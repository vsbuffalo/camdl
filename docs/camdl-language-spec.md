# The camdl Language Specification

**Version:** 0.3-draft **Date:** 2026-07-17

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

**Model + layered configuration.** A `.camdl` file is structurally stable across
all analyses — forward simulation, calibration, scenario comparison, and
forecasting all use the same file with different layered configuration on top.
Two shapes are both first-class:

- A **structural skeleton** that declares parameter names, kinds, and dimensions
  only; values are supplied externally via TOML, CLI flags, or inference
  engines. Useful for "model under inference" / library-style work.
- A **self-contained reproducible model** that, on top of that skeleton, bundles
  the rest of what it takes to read and run the file: `~` prior declarations, an
  `init { ... }` block for initial states, and a `baseline` scenario whose
  `set = { ... }` block supplies concrete parameter values — the one in-file way
  to give a parameter a value. So the file can be handed to a colleague (or
  shipped in a paper supplement) and run end-to-end. This is the preferred form
  for distribution and for the canonical examples shipped with camdl.

**Parameter values are never declared in the `parameters { ... }` block** — it
carries names, kinds, dimensions, and optional priors only (§4, §4.2). Concrete
values live elsewhere: a `params.toml` / `[fixed]` block, a CLI override
(`--param`, `--set`), or — the only in-file form — a named scenario's
`set = { ... }`. The precedence chain is fixed (see `camdl-run-spec.md` §1.3 for
forward simulation and `docs/inference.md` for inference), and it is **not** a
blanket "external beats in-file": an in-file scenario's `set`/`scale` overrides
the external `params.toml` base and any sweep point, and only an explicit CLI
`--param` / `--fixed` outranks the scenario. For a simulation's parameters,
highest precedence first: `--param` CLI flags, then scenario `set = { ... }`,
then sweep points, then the `params.toml` base values. The seed is always a CLI
argument.

What this design preserves — and what the IR's hash discipline enforces — is
that _structural_ model identity (compartments, transitions, observation
projections, intervention semantics) is captured by the **model** level of a
run's factored identity, while _value-bearing_ content lives in later levels:
base params in the **params** level, backend and `dt` in the **config** level,
the scenario delta in the **scenario** level, and inference inputs (priors,
transforms, data, fit config) in the **fit** level (§19). So two analyses that
share a structural model share the same model-level hash even if one bakes in
calibrated values and the other supplies them externally; changing a value is
visible in the params or config level even when the underlying structure didn't
change. The reviewer trying to tell "is this a structural change or a parameter
sweep" reads the level hashes, not the file shape.

**Typed and checked.** Index dimensions, table shapes, compartment arities,
parameter domains, and unit dimensions are compiler-checked with clear error
messages, at compile time rather than simulation time. A **named** index is
resolved by the dimension it names, and a label that is not a dimension of the
thing indexed is rejected (`E332`); a **positional** index is checked against
the levels of whichever dimension holds that slot, so a level from the wrong
dimension is rejected (`E263`).

What the compiler does **not** do is track which dimension an index _variable_
was bound to. With `age = [low, high]` and `risk = [low, high]`, writing
`I[r, a]` where `a` is bound to `age` and `r` to `risk` names an existing cell
and compiles silently. Positional indexing is safe only while the dimensions'
level names are disjoint — which is the concrete reason to prefer named
indexing (§12.1) whenever two axes could ever share a level name.

**No auto-localization.** After stratification, bare compartment names always
refer to the global total (sum over all strata). `S` means "all susceptibles."
`S[child]` means "susceptible children." The compiler never guesses which
stratum you meant. Stratification rules (coupling sugar) handle the
transformation from global to per-stratum formulas mechanically; the user writes
the base model with global names and specifies how dimensions interact.

### 1.1 Syntax Conventions

```camdl
:    structural definition (what something IS)
=    value binding (what something EQUALS)
@    rate expression (how fast, always total propensity)
-->  flow direction
#    comment
#'   doc comment (attaches to the declaration below it)
{ }  block grouping
[ ]  index access and list literals
( )  function arguments
'    unit literal prefix ('days, 'years)
```

#### Doc comments (`#'`)

A line starting `#'` documents the declaration **immediately below it**. Plain
`#` is an ordinary comment that no tool reads; `#'` prose is carried through
compilation and shown by `camdlc inspect`, `camdlc render`, and the parameter
legend in `camdl fit summary`.

A doc comment attaches to a **declaration**, never to a block keyword. Inside a
block that means the member, not the `{`:

```camdl
compartments {
  #' fully susceptible
  S,
  #' confirmed and still transmitting — the infectious dwell ends here
  C
}
```

Writing `#'` above `compartments {` is a syntax error, because a block has no
declaration for it to describe. The sites that take one are: `compartments`
members, `parameters` members, `dimensions` entries, `transitions`,
`observations` streams, `quantities`, and top-level `let` bindings.

Whether `C` means "confirmed and still transmitting" or "confirmed, isolated,
terminal" decides whether the `I` dwell is the effective infectious period, and
so the generation interval, and so the R0 you infer from a given growth rate.
That is a scientific fact about the model, and `#'` is where it lives.

Two tags are recognised inside a doc comment; anything else is `E111`.

```camdl
#' basic reproduction number
#' @symbol R_0
#' @ref Anderson & May 1991, ch. 6
R0 : positive in [0.5, 20]
```

`@symbol` overrides the symbol `camdlc render` prints for that name in the
LaTeX and JSON projections — useful when the model identifier is spelled for
code (`Conf`) and the paper spells it `C`. `@ref` records a citation or case
definition. Both are optional, and either may appear without prose.

A `let` binding is the highest-value site: a derived quantity is where a
modelling assumption hides. `let N[p] = S[p] + E[p] + I[p] + C[p]` shows its
arithmetic but not whether it is the total population or the
force-of-infection denominator, and those are different models.

```camdl
#' total population per patch — the FOI denominator
let N[p in patch] = S[p] + E[p] + I[p] + C[p]
```

Which surfaces a `let`'s prose reaches depends on whether the binding becomes a
model entity. A **typed** `let` with a **constant** body is a fixed parameter —
`let omega : rate = 0.01` is indistinguishable, once compiled, from the same
value declared in `parameters { }` — so its doc joins the parameter legend in
`camdl fit summary` alongside the rest. Every other `let` is inlined and has no
entity of its own; its prose lives in `camdlc inspect --let <name>`. `@symbol`
applies either way, and applies everywhere the name is typeset, not only in the
definition itself.

---

## 2. Time Unit and Dimensional Types

```camdl
time_unit = 'days
```

All rates and durations are normalized to this unit at compile time.

`time_unit` must be a **duration** unit: `'days`, `'weeks`, `'months` or
`'years`. A rate (`'per_day`), `'count` or `'ratio` has no mapping to a length
of time and is rejected with **E228** at the declaration. (`'months` and
`'years` carry the additional anchored-mode restriction described in §2.2.)

### 2.1 Unit Literals

Unit literals are distinguished from identifiers by the `'` prefix:

```camdl
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
`'per_week`, `'per_month`, `'per_year`, `'count`, `'ratio`. `'count` carries
dimension P (population); `'ratio` is dimensionless. Both are used on table
cells (§2.5) and on parameter declarations (§4.1.1) where the dim checker needs
a tier-3 hint that doesn't fit the time-or-rate axis.

Conversions: 1 'week = 7 'days, 1 'month = 365.2425/12 'days ≈ 30.4369, 1 'year
= 365.2425 'days. Proleptic-Gregorian throughout; matches `rata_die` and
`rust/crates/ir/src/caltime.rs::days_per_unit` (the shared conversion
authority).

#### Exact vs Calendar duration kinds

Duration unit literals split into two kinds, distinguished by whether their
magnitude is a constant or an affine average over the calendar:

| Unit literal        | Kind     | Why                                             |
| ------------------- | -------- | ----------------------------------------------- |
| `'days`, `'weeks`   | Exact    | constant day count (1, 7) — invertible          |
| `'months`, `'years` | Calendar | average-length (≈ 30.4369, ≈ 365.2425) — affine |

The dimensional checker tracks this as a **one-bit refinement on `[T]`**
(`Exact <: Calendar` on the subtype lattice). The refinement propagates through
arithmetic by least upper bound — `Exact ± Exact` stays Exact; any expression
touching a `'months`/`'years` literal becomes Calendar.

The refinement is load-bearing in **anchored mode** (when `origin =
date(...)`
is declared): a `Calendar`-classified duration cannot be added to or subtracted
from an `Instant` (**E321**), because calendar months/years are non-invertible
relative to a date. Use `add_calendar_months` / `add_calendar_years` (§2.3) for
calendar-exact stepping, or an explicit `'days` literal for an affine offset.

In **unanchored mode** (no `origin`) the refinement is inactive: no calendar
reference exists, so `5 'months` as an affine span is fine, including in rate
expressions, table values, and parameter bounds. The dacca SIRS configuration
(`time_unit = 'months`, `beta : rate
'per_month`, `6 'months` durations) is
entirely composed of calendar-named _affine_ constructs and works unchanged.

See [`docs/dates.md`](dates.md) and the typed-time proposal
([`docs/dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md`](dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md))
§3 for the full statement of the rule, the LUB propagation table, and the
classifier-from-leaves invariant.

### 2.2 Dimensional Type System

Durations and rates are **distinct types**. The compiler tracks dimensions
through expressions and rejects mismatches:

```camdl
'days     — dimension: time
'per_day  — dimension: 1/time (rate)
```

Valid operations:

```camdl
5 'days + 3 'days         → 8 'days (time + time = time)
1 / (14 'days)            → rate (1/time)
0.1 'per_day * 100        → 10 'per_day (rate × scalar = rate)
0.1 'per_day * 5 'days    → 0.5 (rate × time = dimensionless) ✓
```

Invalid operations:

```camdl
5 'days + 0.1 'per_day    → ERROR: cannot add time and rate
5 'days * 3 'days         → ERROR: time² has no meaning in this system
```

Mixed-unit values in tables that aren't compatible are compile errors.
Dimensionless zero (`0.0`) is compatible with any unit context.

### 2.2.1 Transition Rate Dimensional Analysis

The compiler checks that every transition rate expression has dimension
**P·T⁻¹** (population per unit time). This catches the most common modeling bug:
writing a per-capita rate where a total propensity is needed.

The checker infers dimensions from:

- Compartment references (`Pop`, `PopSum`) → dimension **P**
- `Time` → dimension **T**
- Parameter types: `rate` → **T⁻¹**, `probability` → **1**, `count` → **P**
- Unit literals: `'days` → **T**, `'per_day` → **T⁻¹**
- Arithmetic rules: multiplication adds exponents, division subtracts

```camdl
# ✓ Correct: beta:T⁻¹ * S:P * I:P / N:P = P·T⁻¹
infection : S --> I @ beta * S * I / N

# ✗ Error E300: beta:T⁻¹ * I:P / N:P = T⁻¹ (missing S)
infection : S --> I @ beta * I / N
#   error[E300]: transition 'infection' rate has wrong dimension
#     expected: P·T⁻¹ (population-level rate)
#     got:      T⁻¹ (per-capita rate)
```

Parameters with `kind = positive` or `kind = real` have unknown dimension — the
checker infers it from context. If inference is ambiguous, use a `[dim]`
annotation (see §4.1.1). If a parameter is used inconsistently across
transitions, the compiler emits E303.

Additional checks:

- **E301**: argument to `exp()` / `log()` must be dimensionless
- **E302**: addition/subtraction of mismatched dimensions
- **E304**: `sqrt()` of odd-exponent dimension
- **E305**: balance expression must have dimension P
- **E306**: ODE derivative must have dimension P·T⁻¹
- **E308**: overdispersion σ² must be dimensionless
- **E309**: forcing `lag` must be a duration (dimension T)

Disable with `--no-dim-check` if a false positive is encountered (and file a
bug).

### 2.2.2 Phenomenological dimensional escape: `unchecked_dim`

Some formulations intentionally break dimensional homogeneity. The canonical
case is the He et al. (2010) α-mixing term `(I + ι)^α` with non-integer `α`:
`P^0.976` has no well-defined dimension, but the formulation is empirically
validated and widely used. These are legitimate modelling choices, not bugs.

`unchecked_dim(expr, dim = NAME, reason = "…")` asserts that the wrapped
expression has the named dimension without the dim-checker verifying the
assertion. Surrounding rate expressions continue to dim-check normally — the
escape is narrow and visible.

<!-- camdl-doctest-preamble: unchecked-dim
compartments { S, E, I, R }
parameters {
  gamma : rate
  iota  : count
  alpha : positive
}
forcing {
  beta : sinusoidal 'per_day {
    baseline  = 0.5
    amplitude = 0.0
    period    = 365.25 'days
    phase     = 0 'days
  }
  pop : sinusoidal 'count {
    baseline  = 100000.0
    amplitude = 0.0
    period    = 365.25 'days
    phase     = 0 'days
  }
}
-->

```camdl preamble=unchecked-dim
transitions {
  infection : S --> E
    @ beta(t) * unchecked_dim((I + iota)^alpha,
                              dim = population,
                              reason = "He et al. 2010 α-mixing exponent")
             * S / pop(t)
}
```

Valid `dim` names: `dimensionless`, `population`, `time`, `rate`,
`population_rate`, `per_population`. `reason` is required — a string documenting
the assertion's legitimacy. The wrapper compiles to `Ir::UncheckedDim` and is
transparent at runtime (identity over `inner`).

**Choosing the asserted dimension.** The assertion must make the _surrounding_
expression typecheck. For the He case above: `β(t)` is rate (T⁻¹), `S / pop(t)`
is dimensionless, so `unchecked_dim(…)` must absorb the population-exponent for
the full rate to be P·T⁻¹. Hence `dim = population`. A common mistake is
asserting `dimensionless`, which leaves the full rate at T⁻¹ and triggers
downstream dim errors.

Use sparingly — `unchecked_dim` should feel like an escape hatch, not a normal
tool. When a build has `unchecked_dim` sites, each should be reviewed in code
review for legitimacy.

References for the canonical case:

- He, Ionides, & King (2010). _J. R. Soc. Interface_ 7(43): 271–283.
  doi:10.1098/rsif.2009.0151.
- Bretó, He, Ionides, & King (2009). _Annals of Applied Statistics_ 3(1):
  319–348. doi:10.1214/08-AOAS201.

### 2.3 Date Literals

The `date("YYYY-MM-DD")` expression converts an ISO 8601 date to a float offset
from the model's declared `origin` date, in the model's `time_unit`:

```camdl preamble=sir-basic
origin = date("2019-01-01")   # top-level declaration (optional)

simulate {
  to = date("2021-06-30")     # 911 days from origin (in time_unit = 'days)
}
```

`date(...)` uses proleptic Gregorian calendar arithmetic. The result is exact
(integer day count) when `time_unit = 'days`, and divided by the appropriate
factor for other units (e.g., 365.2425 for years — the Gregorian average year,
matching §2.1).

**E220.** Using `date(...)` without a top-level `origin = date(...)` declaration
is a compile error:

```camdl
# ERROR E220: date("2021-06-30") used but no 'origin' declared
simulate { to = date("2021-06-30") }
```

The `origin` value is stored in the IR as both the ISO string
(`"origin": "2019-01-01"`) and a compiler-derived integer day number
(`"origin_rata_die": …`) that the runtime reads without re-parsing. It does not
affect simulation dynamics — it is purely a coordinate reference for converting
calendar dates to simulation time.

Calendar support extends beyond `date()` literals: **observation data may use an
ISO-date time column** (auto-converted via `origin` + `time_unit`), output can
be rendered back as dates (`simulate --dates`, and `instant`-kind estimands in
`fit
summary`), and the **`instant` / `duration` parameter kinds** carry
dimension `[T]`. **See [`docs/dates.md`](dates.md) for the complete, canonical
treatment** of dates across the DSL, data loading, and output.

#### Calendar-arithmetic primitives and `origin` as a referenceable Instant

In **anchored mode** (when `origin` is declared), two compile-time primitives
step a date by calendar months or years, available in DSL constant positions
only:

```camdl
add_calendar_months(d, n)   # Instant × Int → Instant
add_calendar_years(d, n)    # Instant × Int → Instant
```

`d` is any compile-time-constant Instant — a `date(...)` literal, the reserved
`origin` identifier, or a nested `add_calendar_*` call. `n` is a compile-time
integer. The algorithm is proleptic-Gregorian `(year, month, day)` arithmetic
with **month-end clamping**:
`add_calendar_months(date("2020-01-31"), 1) = date("2020-02-29")` (leap),
`add_calendar_months(date("2021-01-31"), 1) = date("2021-02-28")`. These
functions never touch the `30.4369` average-month factor — they're the only
correct way to step a date by calendar months/years. In unanchored mode, a call
to either primitive is **E327**.

`origin` is **reserved in anchored mode as a referenceable compile-time-constant
`Instant`**, usable wherever a constant Instant is accepted —
`simulate { from = origin }`, `at [origin, ...]` schedules,
`add_calendar_months(origin, 6)`, `origin + 90 'days`,
`let landmark = origin + 90 'days`. Not usable inside rate expressions or any
compartment-state context (it's a compile-time constant, not a runtime value).
In unanchored mode, a reference to `origin` is **E327**.

A `date_range(...)` compile-time generator produces a list of Instants from a
start, an end-or-count, and a cadence — see [`docs/dates.md`](dates.md) for the
surface and
[`docs/dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md`](dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md)
§4 for the full signature and diagnostics.

### 2.4 Three tiers of dimensional information

Dimensional information in a model can be declared at three levels of
specificity. Each tier carries strictly more information than the next, and each
is the right tool for a distinct class of declaration. Understanding the
hierarchy makes the rest of §2 and §4–7 fit together.

| Tier                      | Syntax                                                                    | Carries                        | Use when                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| 1. **Kind keyword**       | `rate`, `probability`, `count`, `positive`, `real`, `instant`, `duration` | Dimension (inferred from kind) | Common parameter cases (the 99% case). `instant`/`duration` are time-typed (`[T]`) — see [`dates.md`](dates.md). |
| 2. **Bracket annotation** | `[T]`, `[T^-1]`, `[P]`, `[P/T]`, `[1]`                                    | Dimension only                 | Kind keyword is under-determined (`real`, `positive`).                                                           |
| 3. **Unit literal**       | `'days`, `'years`, `'per_day`, `'per_year`, `'count`, `'ratio`            | Dimension **+ scale**          | Concrete numeric data with a known real-world scale.                                                             |

**The tiers are complementary, not redundant.** They answer different questions
about a value:

- **Bracket annotations** answer _"what dimension does this parameter have?"_ —
  for values supplied later (at fit time via `--params`, from priors, from
  inference). The compiler type-checks dimensions without knowing scales. Scale
  is implicit: everything is in the model's `time_unit`.

- **Unit literals** answer _"what dimension AND what real-world scale does this
  value have?"_ — for concrete data (`time_unit` declaration, duration literals
  like `5 'years`, table entries, data loaded via `read()`). The compiler
  type-checks dimensions AND normalises scale to the model's `time_unit`.

You can't collapse tiers 2 and 3 into one syntax. Brackets can't carry scale:
`[T]` says "time-dimension," not "days" vs. "years." A user supplying
`beta = 0.3` via `--params` has agreed to provide it in model time units — the
compiler doesn't need to convert. Unit literals can't be applied to unknowns:
you can't write `beta :
positive 'per_day` when `beta`'s value will be drawn
from a prior, because the prior samples directly in model time units.

**Subtyping.** Every unit literal decomposes to (dimension, scale). The
dimension half plays the same type-checking role as a bracket; the scale half
drives scale normalisation. A unit literal is strictly more informative than a
bracket of the same dimension:

```
'years       ⊂    [T]     ⊂   (any time-dimensioned value)
(dim+scale)       (dim)         (unconstrained)
```

**Kind keywords are a fourth, convenience layer on top** of the three tiers.
They name common dimension patterns (`rate` ≡ `[T^-1]`, `probability` ≡ `[1]`
with `[0,1]` domain, `count` ≡ `[P]`) so 99% of parameter declarations avoid
bracket notation entirely. Use brackets only when the kind keyword is
under-determined (`real`, `positive`); use unit literals only where concrete
numeric values are attached. This section (§2.4, §6.1) shows tier-3 use on table
values; §4.1.1 shows tier 2 on parameter declarations; §7 (Forcing) requires
tier 3 on every forcing declaration.

**`'ratio` vs `probability`**: both are dimensionless, but they're not synonyms.
`'ratio` (tier 3) is the **unbounded** dimensionless case — a multiplier that
could be 0.7, 1.3, 50, used for seasonal forcings, school-term indicators,
reporting multipliers. `probability` (tier 1, parameter kind) is the **bounded**
dimensionless case — values constrained to [0, 1] with an automatic logit
transform for inference. Reach for `'ratio` when you want an arbitrary
multiplier; reach for `probability` when you want a bounded probability
parameter.

### 2.5 Table Unit Annotations

Tables carry a single unit for all values:

```camdl
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

```camdl
compartments { S, E, I, R }
```

Each is an integer-valued population count. For continuous state:

<!-- camdl-doctest-preamble: comp-real
parameters {
  beta  : rate
  gamma : rate
  xi    : rate
  delta : rate
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
ode {
  W = xi * I - delta * W
}
-->

```camdl preamble=comp-real
compartments {
  S, I, R,
  W : real       # continuous-valued (environmental reservoir)
}
```

After stratification, compartments gain index dimensions (see §5). Access is
always via explicit indexing: `S[child]`, `S[child, female]`, or bare `S` (= sum
over all strata).

---

## 4. Parameters

```camdl
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

Parameters are **declared** here — names, kinds, and dimensions only. Concrete
values are **never** specified in the model file; they are supplied externally
via CLI flags, a `--params` TOML, or inference engines (§4.2).

### 4.1 Parameter Types

```
rate        : ≥ 0, dimension 1/time. Default transform: log.
probability : ∈ [0, 1], dimensionless. Default transform: logit.
positive    : > 0, dimensionless. Default transform: log.
count       : dimension P (a population count). Integrality is not enforced —
              `let iota : count = 1e-6` compiles; the type carries the [P]
              dimension, not an integer constraint.
real        : unconstrained (default if omitted).
instant     : dimension [T], absolute time. Renders as a date in anchored
              mode (requires origin). Negative lower bounds allowed.
duration    : dimension [T], a span. Renders as a span.
```

There is no `ivp` ("initial-value parameter") kind. Stochastic initial states
are expressed through the `init {}` block (§15) and made estimable by the
inference engine's `ic_free` setting — not a parameter kind.

Types enable: validation of supplied values, default inference transforms, and
dimensional analysis of rate expressions.

**On `instant` vs `duration` in dimensional checking.** Both carry dimension
`[T]`. In **anchored mode** (top-level `origin` declared), `instant` is the
Instant side of the typed-time torsor: it can be added to / subtracted from an
`Exact` duration (§2.1), and an `Instant − Instant` is itself an Exact duration.
In **unanchored mode** the torsor refinement is inactive — both kinds behave as
plain `[T]`-dimensioned scalars and are interchangeable in expression
arithmetic. See the typed-time proposal §1.1 for the formal "classification
synthesised per occurrence from its leaves" invariant: a **reference** to a
`duration`-kind parameter is always classified `Exact`, even when the
parameter's _bound_ is spelled in `'months` (e.g.
`delay : duration in [1 'months, 6 'months]`). The bound's month-spelling is a
compile-time-evaluable length, not a step-from-a- date, and never leaks
`Calendar` to uses of the parameter.

### 4.1.1 Dimension Annotations

When a parameter's type doesn't fully determine its dimension (e.g., `positive`
could be a rate, a count, or dimensionless), you can add an explicit dimension
annotation:

```camdl
parameters {
  beta      : rate                    # dimension T⁻¹ (inferred from type)
  gamma     : rate                    # T⁻¹
  amplitude : real [1]                # explicitly dimensionless
  mu        : positive [1/T]          # per-capita rate
  coupling  : positive [P/T]          # population-level rate
  R0        : positive [1]            # dimensionless
}
```

The annotation goes in square brackets after the type, before optional bounds.
Supported dimension literals:

| Annotation | Dimension         | Domain name            |
| ---------- | ----------------- | ---------------------- |
| `[1]`      | dimensionless     | probability, ratio, R₀ |
| `[P]`      | population        | count                  |
| `[T]`      | time              | duration               |
| `[1/T]`    | T⁻¹               | per-capita rate        |
| `[T^-1]`   | T⁻¹ (alternate)   | per-capita rate        |
| `[P/T]`    | P·T⁻¹             | population-level rate  |
| `[P*T^-1]` | P·T⁻¹ (alternate) | population-level rate  |

When present, the annotation overrides type-based inference. If the annotation
conflicts with how the parameter is used in rate expressions, the compiler emits
a dimension error.

Annotations are optional. The compiler infers dimensions from context for most
models — annotations are only needed when inference is ambiguous (reported as
info I300).

### 4.1.2 Unit literal on `positive` / `real`

The dimension-under-determined kinds — `positive` and `real` — also accept an
optional **tier-3 unit literal** (§2.4) in place of a bracket annotation:

```camdl
parameters {
  tau    : positive 'ratio    in [0.001, 3.0]    # dimensionless CV (= positive [1])
  iota   : positive 'count    in [0.0, 50.0]     # a count seed   (= positive [P])
  importn: positive 'per_year in [1e-4, 0.1]     # per-time rate  (= positive [T^-1])
  amp    : real 'ratio                            # signed multiplier
}
```

A unit literal here is read for its **dimension only** — the dimension half of
the (dimension, scale) pair a unit literal carries (§2.4). The scale half plays
no role on a parameter: a parameter's value is always supplied in the model's
`time_unit`, so `'per_year` and `'per_day` set the same dimension (T⁻¹) on the
declaration. The literal is therefore exact sugar for the corresponding bracket
annotation: `positive 'ratio` ≡ `positive [1]`, `positive 'count` ≡
`positive [P]`, `positive 'per_year` ≡ `positive [T^-1]`. The two notations are
interchangeable; use whichever reads more naturally for the quantity.

The annotation fixes the parameter's dimension exactly as a bracket does — so it
resolves the I300 a bare `positive`/`real` would emit in a dimension-determined
slot, **and** turns a dimensional misuse into a hard error. A `tau : positive
'ratio` added to a population (`I + tau`) is now an E302, not a swallowed I300;
that is the point — a determined dimension catches the bug a `positive` would
hide.

Two restrictions keep the surface honest:

- A unit literal is only meaningful on `positive` and `real`. On a kind whose
  dimension the keyword already fixes (`rate`, `probability`, `count`,
  `instant`, `duration`), it is rejected with **E281** — drop the literal; the
  kind already carries the dimension.
- A unit literal and a `[dim]` bracket annotation may not both appear on one
  declaration (they would be redundant or contradictory): **E282**. Use one or
  the other.

### 4.2 External parameter values

Parameter values are **never** specified in the `parameters { ... }` block. The
block declares names and types only; concrete values are supplied at runtime — a
named scenario's `set = { ... }` block is the one in-file exception:

```bash
# Single flat TOML file
camdl simulate model.ir.json --params base.toml

# Layered overrides (later files win)
camdl simulate model.ir.json --params base.toml --params patch.toml

# Single value override
camdl simulate model.ir.json --param gamma=0.1

# Per-stratum override (indexed params)
camdl simulate model.ir.json --param-vec R0=r0_posterior.tsv
```

The TOML format supports both flat and sectioned forms (see §21).

Because values live outside the model, a `--params` TOML is the right home for
any externally-computed scalar — a demographic rate from a pipeline, a fixed
constant from the literature. Have preprocessing **generate** that TOML so the
model consumes the pipeline output directly (single source of truth); do not
reach for a `tables {}` entry to read a scalar. Tables are indexed data
(covariates or feature, not observation data which would be in an `observation`
block) and require at least one dimension (§6).

### 4.3 Indexed Parameters

Parameters may be declared with one or more dimension indices, creating one
scalar parameter per stratum (or, for several indices, per cell of the cartesian
product):

```camdl
parameters {
  gamma               : rate
  N[patch]            : positive   # expands to N_urban, N_rural, ...
  R0[patch]           : positive   # expands to R0_urban, R0_rural, ...
  mu[village, season] : rate       # expands to mu_kwaru_wet, mu_kwaru_dry, ...
}
```

A multi-index parameter is a design matrix: `mu[village, season]` over
`village = [kwaru, ajura]`, `season = [wet, dry]` expands to the four cells
`mu_kwaru_wet`, `mu_kwaru_dry`, `mu_ajura_wet`, `mu_ajura_dry`, each an
independent scalar with the declared kind, bounds, and prior. Names mangle
`<base>_<level1>_<level2>_…` in declaration-dim order. A repeated axis
(`mu[village, village]`) is an error (E331); an unknown or empty index dimension
is E330.

Each index must refer to a declared `stratify` dimension. In expressions,
indexed parameters are accessed with `[index]` (all axes, in declaration order):

```camdl
let C[v in village, s in season] = mu[v, s] * gamma   # mu[v,s] → Param("mu_kwaru_wet") etc.
```

For a single index:

```camdl
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

```camdl
let urban = 1.0   # W103: let binding 'urban' shadows stratum value 'urban'
                  #   in dimension 'patch'. This is allowed but consider renaming.
```

**Consistent indexed syntax everywhere.** The `N0[urban]` form works in all
contexts where a parameter name can appear: expressions (§9.6), init blocks
(§15.2), and scenario set/scale blocks (§17.1). The compiler always mangles to
`N0_urban` in the IR.

**IR representation.** Indexed parameter declarations expand to flat scalar
parameters:

```
N[patch] : positive  →  { name: "N_urban",  value: null }
                         { name: "N_rural",  value: null }
```

**Runtime override.** Use `--param-vec PREFIX=FILE` to supply per-stratum values
at runtime (see §21):

```bash
camdl simulate model.ir.json --param-vec R0=/tmp/r0_posterior.tsv
```

### 4.4 Parameter Bounds

An optional `in [lo, hi]` clause constrains the parameter's valid range:

```camdl
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

```camdl
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

<!-- camdl-doctest-preamble: partial-strat
compartments { S, E, I, R }
parameters {
  beta  : rate
  sigma : rate
  gamma : rate
}
let N = S + E + I + sum(m in immunity, R[m])
transitions {
  infection : S --> E @ beta * S * I / N
  progress  : E --> I @ sigma * E
  recovery  : I --> R[natural] @ gamma * I
}
-->

```camdl preamble=partial-strat
dimensions { immunity = [natural, vaccine] }
stratify(by = immunity, only = [R])
```

After this, S/E/I have dimensions `[age, sex]` but R has `[age, sex, immunity]`.

> **See also:** for sequential transitions across dimension levels (aging across
> age bins, Erlang sub-stages of a compartment), use the `consecutive(dim)`
> index binding documented in §9.4. Don't enumerate level-by-level transitions
> by hand — `[(a, a_next) in consecutive(age)]` generates them all from one
> declaration.

### 5.0 Two kinds of axis

A compartment's axes do not all come from the same place, and they do not all
behave the same way. Both terms below are normative.

**Population strata** are declared in `dimensions {}` — age, patch, risk group,
vaccination status. They partition individuals.

**Residence structure** is created by the compiler from `via erlang` (dwell-time
stages) or `via hyper_erlang` (exit pathways). It describes _how long_ an
individual stays in a compartment and _by which exit_ they leave — not who they
are.

|                               | population stratum          | residence structure                                    |
| ----------------------------- | --------------------------- | ------------------------------------------------------ |
| how it arises                 | declared in `dimensions {}` | generated by `via` lowering                            |
| what it denotes               | a partition of individuals  | position in a dwell-time law, or exit pathway          |
| indexable?                    | yes — `I[child]`            | no — never named by the modeller                       |
| bare name collapses it?       | yes                         | yes                                                    |
| data can be stratified by it? | yes — an `age : dim` column | no — no measurement distinguishes stage 2 from stage 3 |
| trajectory columns?           | yes — `I_child`             | yes — `I_child_s1`                                     |
| counts toward index arity?    | yes                         | **no**                                                 |

The rule that falls out:

> **You may omit residence structure from an index; you may not omit a
> population stratum.**

So on a compartment with axes `[age, __recovery_stage]`, `I[b]` is a complete
index — it fixes the age band and pools the stages — even though `I[b]` on an
`[age, patch]` compartment is a partial index and `E287`. The justification is
representational, not epidemiological: `via` creates several cells for **one
declared compartment**, and a modeller who writes
`onset : E --> I via erlang(stages = 3, …)` declared one `E`.

Two consequences worth stating:

- **Hand-rolled staging is a population stratum.** `dimensions { latent_stage
  = … }` plus `stratify(by = latent_stage, only = [E])` is declared structure,
  so it is indexable and counts toward arity. Public model structure is public.
- **`via hyper_erlang` creates no dimension at all.** It emits flat per-branch
  cells (`I__fatal__1`, …), so nothing keyed on a dimension can describe it.
  Those generated _compartment_ names are intentionally visible — they are
  trajectory columns and scenario-referenceable transition names.

### 5.1 Indexing Rules

**Positional indexing** (declaration order of stratify blocks):

```camdl
S[child]                # first dimension = age
S[child, female]        # age, then sex
S                       # bare = sum over ALL strata (always global)
S[child]                # if S has [age, sex]: ERROR (E287) — a partial index
                        #   has no defined cell; the bare name S sums, but
                        #   S[child] neither sums nor picks a cell
```

**Named indexing** (explicit dimension labels, any order):

```camdl
S[age = child]                    # equivalent to S[child]
S[sex = female, age = child]      # order doesn't matter
S[age = child, sex = female]      # a non-first dim named; every dim still specified
```

Named indexing is useful when a compartment or transition has multiple
dimensions and you want to index a non-first dimension. The compiler resolves
named indices to positional and validates dimension membership.

Positional and named indexing can be mixed: `S[child, sex = female]` is valid
(first positional = age, second named = sex). But for clarity, use one style
consistently.

> **Status (gh#459).** The two paragraphs above describe the specified
> behaviour; neither is implemented yet. The compiler discards the dimension
> label and lowers named indices by **source order**, so it performs no
> membership check and order does matter:
>
> ```text
> # dimensions declared [age, patch]
> S[age = child, patch = north]   → resolves to S_child_north   (correct, by luck of ordering)
> S[patch = north, age = child]   → error[E100]: undeclared name 'S_north_child'
> S[age = north, patch = child]   → error[E100]: undeclared name 'S_north_child'
> #                                 ^ no membership error, though `north` is not an `age` level
> ```
>
> Until gh#459 lands, write named indices in **declaration order**, and treat
> the labels as documentation rather than as checked constraints. Where two
> dimensions share level names the wrong order can silently name a *different
> existing cell*.

**Omitting ALL dimensions sums over them; a partial index is an error.** The
compiler knows each compartment's arity and checks every access.

**In rate expressions** (right of `@`):

- The **bare name** (no brackets) sums over *every* dimension — a global scalar
  read. `R` when R has `[age, immunity]` means
  `R[child, natural] + R[child, vaccine] + R[adult, natural] + R[adult, vaccine]`.
- A **fully-indexed** access resolves to one cell: `R[a, natural]`.
- A **partial index** — some dimensions supplied, some dropped — is an
  **error** (`E287`). `R[a]` when R has `[age, immunity]` has no defined cell:
  the compiler cannot tell whether you meant the `[a, natural]` cell, the
  `[a, vaccine]` cell, or their sum. To sum over a dimension while fixing
  another, marginalize it **explicitly**: `sum(m in immunity, R[a, m])`.

**In stoichiometry** (left of `@`, source/destination of `-->`): **all
dimensions of the compartment must be specified.** You cannot write into a
marginal — the compiler must know exactly which cell gains or loses an
individual.

```camdl
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
[i in age]                     # binds i to iterate over age values
sum(j in age, expr)            # binds j, sums expr over age values
sum(j in age, k in patch, e)   # binds several axes in one sum (§8.2)
```

The `in dim` clause makes the dimension explicit. The compiler tracks which
dimension each variable belongs to.

### 5.3 Partial Stratification in Expressions

When compartments have different dimensions due to `only = [...]`, bare names
and indexed access follow the same rules — but the compiler resolves them
per-compartment based on each compartment's actual arity.

Example: `E` has `[age, latent_stage]`, `S` has `[age]`.

```camdl
S + E                          # both are global sums: PopSum(all S) + PopSum(all E). Valid.
S[a] + sum(s in latent_stage, E[a, s])
                               # S in age=a + E in age=a (summed over latent_stage). Valid.
S[a] + E[a, e1]                # S in age=a + E in age=a and stage=e1. Valid.
S[a] + E[a]                    # ERROR (E287): E has [age, latent_stage] — `E[a]` is a partial index.
S[a, e1]                       # ERROR: S has no latent_stage dimension.
```

The bare-name-sums rule applies per-compartment: `E` (no brackets) sums over
both `age` and `latent_stage`, while `S` sums over `age` only. But once you
index *any* dimension you must index *all* of them, per compartment: `S[a]` is
fully resolved because `S` has only `[age]`, whereas `E[a]` is a partial index
(it drops `latent_stage`) and is rejected — to fix `age` and sum over
`latent_stage`, write `sum(s in latent_stage, E[a, s])`. The compiler tracks
each compartment's dimensions independently.

---

## 6. Tables

```camdl
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

**Tables are indexed data — they require at least one dimension.** Every table
above is keyed by `age`, `patch`, etc. There is no 0-dimensional (scalar) table:
`read()` loads an indexed array, not a single value. A scalar input — even one
computed by preprocessing (a crude birth rate, an all-age mortality rate) — is a
**parameter**, not a table. Declare it in `parameters {}` and supply its value
via `--params` (§4.2); your preprocessing pipeline can emit that TOML, keeping a
single source of truth without resorting to a dummy 1-element dimension.

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

**Separator: `×` or `*`.** The dimension product is written with `×` (the
Unicode multiplication sign, U+00D7). For hand-authoring without the glyph, the
ASCII `*` is accepted as an exact alias: `age * age` compiles identically to
`age × age` (the separator is purely syntactic — it names the axes, nothing
else). Prefer `×` in committed models for readability; the compiler treats them
the same, and the same equivalence already holds in rate expressions, where `×`
and `*` both mean multiplication.

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

```camdl
tables {
  pop : patch = read("data/lga_pop.tsv")
}
```

The type signature declares how many index columns there are (one per dimension
listed). The remaining column(s) are value(s). Column names in the file are for
human readability — the compiler uses **positional mapping** from the type
signature.

**Extension determines separator:** `.tsv` → tab, `.csv` → comma, anything else
→ compile error. The first non-comment row is the required header. Lines that
begin with `#` (and blank lines) are skipped wherever they appear, so a file may
carry leading provenance comments (source URL, fetch date) above the header —
the usual convention for committed reference data.

**Sparse tables** use `default = value` to fill index combinations missing from
the file:

```camdl
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

```camdl
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

<!-- camdl-doctest-preamble: tables-demo
compartments { S, I, R }
dimensions {
  patch = [kano_dala, borno_maiduguri, borno_gwoza]
}
stratify(by = patch)
parameters {
  beta  : rate
  gamma : rate
}
let N[p in patch] = S[p] + I[p] + R[p]
transitions {
  infection[p in patch] : S[p] --> I[p] @ beta * S[p] * I[p] / N[p]
  recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]
}
-->

<!-- camdl-doctest-data: data/demographics.tsv
patch	pop	init_sus
kano_dala	485000	0.88
borno_maiduguri	345000	0.91
borno_gwoza	78000	0.79
-->

```camdl preamble=tables-demo
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
self-contained — no file references at runtime. The canonical IR format is
compact one-element-per-line JSON (a `--pretty` variant exists for inspection);
there is no binary format.

### 6.6 Parameterized Table Entries

Inline table values can be parameter names or arithmetic expressions, not just
numeric literals:

<!-- camdl-doctest-preamble: table-bsex
compartments { S, I, R }
dimensions { sex = [female, male] }
stratify(by = sex)
parameters {
  beta_mf : rate
  beta_fm : rate
  gamma   : rate
}
let N_local[s in sex] = S[s] + I[s] + R[s]
transitions {
  infection[s in sex] : S[s] --> I[s] @ S[s] * sum(t in sex, B_sex[s,t] * I[t] / N_local[t])
  recovery[s in sex]  : I[s] --> R[s] @ gamma * I[s]
}
-->

```camdl preamble=table-bsex
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

Named time-dependent forcing functions, usable in rate expressions. Six
built-in types cover real-world needs:

- `sinusoidal` — smooth seasonal forcing
- `periodic` — repeating step function (day-of-week, month-of-year effects)
- `piecewise` — non-repeating step function (policy changes, campaign windows)
- `interpolated` — data-driven time series (empirical covariates)
- `fourier` — finite Fourier series with estimable cos/sin harmonic pairs
  (`period`, `harmonics = [[a1, b1], [a2, b2], ...]` — each harmonic is a
  2-element list), for smooth periodic forcing richer than a single sinusoid
  (gh#59)
- `periodic_spline` — periodic B-spline with uniform knots (`period`, `n_basis`,
  optional `degree` = 3), for flexible smooth seasonality (gh#59)

<!-- camdl-doctest-preamble: forcing-demo
compartments { S, I, R }
parameters {
  beta       : rate
  gamma      : rate
  alpha      : probability
  phi_season : duration
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
-->

<!-- camdl-doctest-data: data/nga_pop.csv
year,total_pop
2000,1000000
2001,1010000
2002,1020000
2003,1030000
-->

```camdl preamble=forcing-demo
forcing {
  seasonal : sinusoidal 'ratio {
    amplitude = alpha           # can reference parameters (for inference)
    period    = 365.25 'days
    phase     = phi_season      # convention: time from t=0 to peak, in model time_unit
    baseline  = 1.0
  }

  lockdown : piecewise 'ratio {
    breakpoints = [60 'days, 120 'days]
    values      = [1.0, 0.3, 1.0]
  }

  pop_trend : interpolated 'count {
    data      = "data/nga_pop.csv"
    time_col  = "year"           # quoted file column (outside the model)
    value_col = "total_pop"      # quoted file column (outside the model)
    method    = linear           # bare enum: linear | constant | spline
  }

  reporting_dow : periodic 'ratio {
    period = 7 'days
    values = [1.2, 1.1, 1.0, 1.0, 0.9, 0.8, 0.7]
  }

  # Range-based periodic: specify active ranges instead of listing values.
  # step = bin width; on = list of lo:hi ranges where the value is 1.0.
  # Bins outside the ranges are 0.0. The compiler generates the values array.
  school : periodic 'ratio {
    period = 365.25 'days
    step   = 1 'days
    on     = [7 'days : 100 'days, 115 'days : 199 'days,
              252 'days : 300 'days, 308 'days : 356 'days]
  }
}
```

The shape is `NAME : KIND 'unit { … }` for every forcing — the colon-and-block
form is the only one the parser accepts. The tier-3 unit literal is mandatory
(`'ratio`, `'count`, `'per_day`, etc.); see "Required unit literal" below.

Forcing functions compile to `TimeFunc` nodes in the IR. Their arguments can
reference parameters (e.g., `amplitude = alpha`), enabling inference over
function characteristics (e.g., inferring seasonal amplitude).

The `periodic` type supports two forms:

- **Values form:** `values = [1.0, 0.0, 1.0, ...]` — explicit list. Bin width =
  period / len(values). Use for general periodic patterns.
- **Range form:** `step = 1 'days` + `on = [7:100, 115:199]` — binary on/off
  with range literals. The compiler generates the values array. Use for
  calendars with known active periods (school terms, work weeks, campaign
  windows). **In anchored mode** (top-level `origin` declared), bare-numeric
  entries inside `on=[...]` are **E323** — anchored periodic schedules must use
  `date(...)` entries (or, for the rare legitimate "day-offset from origin"
  case, the `--time-format internal-days` opt-in on the data boundary). See
  [`docs/dates.md`](dates.md) and the typed-time proposal §3 (Rule 2 /
  bare-numeric subsection).

Forcing functions are used in rate expressions by name or with explicit `(t)`:

<!-- camdl-doctest-preamble: forcing-school
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
}
let N = S + I + R
forcing {
  school : periodic 'ratio {
    period = 365.25 'days
    values = [0.7, 1.3, 0.7, 1.0]
  }
}
-->

```camdl preamble=forcing-school
transitions {
  infection : S --> I  @ beta * school(t) * S * I / N
  #                          ^^^^^^^^ forcing function reference
}
```

Both `school` and `school(t)` are valid; `school(42)` or other non-`t` arguments
produce an error.

`periodic` is primarily useful in v0.2 reporting pipelines for day-of-week
effects on case reporting.

### Lagged forcing (`lag`)

A forcing often drives the dynamics with a delay: a vectorial-capacity series
shapes transmission only after the mosquito and incubation pipeline, an
intervention's covariate bites days after it is recorded. Declare that delay
once, on the forcing definition, with an optional `lag`:

```camdl
forcing {
  C : interpolated 'per_day {
    data      = "vectorial_capacity.tsv"
    time_col  = "t"
    value_col = "C"
    lag       = 10 'days        # evaluate C at t − 10 days
  }
}
```

### Anchoring a `piecewise` fork to the data

A `piecewise` forcing's `breakpoints` are usually calendar dates. When the fork
is *"where the observed record ends"* — the standard shape for a forecast
scenario — writing that date by hand means re-editing the model on every data
release. A breakpoint may instead be an **observation anchor**, `last_obs` or
`first_obs`, each optionally `±` a constant duration:

```camdl
forcing {
  ramp_control : piecewise 'ratio {
    breakpoints = [last_obs, last_obs + 1 'weeks, last_obs + 2 'weeks]
    values      = [1.0, 0.8, 0.6, 0.4]
  }
}
```

The knots are resolved once per run, from the observation times the run binds,
and each resolved value is printed on stderr. The resolved knots are substituted
into the model the run is content-addressed by, so two data vintages key to
different runs and the store cannot serve one in place of the other.
Two consequences worth stating plainly, because they are behaviour changes
relative to a literal date:

- The fork becomes **fit-config-dependent**. Two fit configurations binding
  different observation streams resolve different knots from the same model
  file.
- A command with **no observed data cannot run the model at all** — a plain
  `simulate` refuses, naming the forcing, unless `--fit` supplies the data.

Anchors are accepted only in `breakpoints`, not in `values` or in any
coefficient list: an anchor is a time, and the knots are the only times a
forcing declares. Same grammar as everywhere else — the anchor must be the whole
entry, the offset must carry a duration unit, and a `'months`/`'years` offset
under an `origin` is refused (see `value_at` above for the full rules and error
codes).

The forcing is then used exactly as any other — `C` or `C(t)` — and resolves to
`C(t − lag)`. `lag` is a property of the forcing as a whole, uniform across all
kinds (`interpolated`, `piecewise`, `periodic`, `sinusoidal`, `fourier`,
`periodic_spline`): every kind is evaluated at the single shifted time.

- **`lag` is a duration** in the model's `time_unit`, unit-aware exactly like
  `period = 365 'days`. It must carry the time dimension; a non-duration `lag`
  (a rate, a count) is a hard error (**E309**).
- **`lag` may reference a parameter** (`lag = tau` where `tau : duration`) — the
  delay can itself be inferred. This is a primary motivation for the feature.
- **`lag = 0` is the identity**, and a forcing declared without `lag` behaves
  exactly as before — no shift.
- The semantics are a **point (Dirac) delay**: the evaluation time is shifted by
  the lag. Distributed-lag kernels (smearing the forcing over a window) are out
  of scope.

A fitted `lag` (an estimated parameter in `lag = tau`) has **no gradient-based
estimation path today**: the compiler emits no `∂forcing/∂lag`, so a `lag`
parameter is rejected under gradient-based PGAS+NUTS with a diagnostic pointing
to the gradient-free estimators. Estimate a `lag` parameter with **IF2** or the
**bootstrap particle filter** (or PGAS with `--no-nuts`); a fixed (literal or
non-estimated) `lag` works under every method.

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

### Required unit literal (tier-3)

**Every forcing declaration must carry a tier-3 unit literal** between the kind
keyword and the block. The literal states both the scale (values are normalised
to the model `time_unit` at expand time) and the dimension (used by the
dim-checker as authoritative — no value-based inference fallback). GH #8.

```camdl
forcing {
  seasonal  : sinusoidal   'ratio    { baseline = 1.0  amplitude = 0.3  … }
  beta_seas : sinusoidal   'per_day  { baseline = beta  amplitude = amp  … }
  school    : periodic     'ratio    { values = [0.7, 1.3, 0.7, …]  … }
  pop       : interpolated 'count    { data = "…"  time_col = "t"  value_col = "pop"  method = linear }
  birthrate : interpolated 'per_year { data = "…"  time_col = "t"  value_col = "rate"  method = linear }
}
```

Omitting the literal is a syntax error (E001). Previously the parser accepted
forcings without annotations and fell back on value-based dimensional inference,
which silently went wrong for `interpolated` (data-file values are always
literal constants → dimensionless) and caused E300 dim-check failures on correct
models. The required-literal design eliminates that class of bug at parse time.

For file-backed `interpolated` forcings, the `time_col` is the model's internal
time axis. Cells may be bare numbers (already in the model `time_unit`) or — in
anchored mode — ISO dates (`YYYY-MM-DD`), which the compiler resolves to internal
time via `origin` + `time_unit` at expand time, the same rule observation-data
time columns and `instant`/`duration` table cells follow (see
[`docs/dates.md`](dates.md)). A date-valued `time_col` in an unanchored model
(no `origin`) is a hard error (E209), never a silent fall-through to zero.

Unit-literal choices for forcings in practice:

- **`'ratio`** — dimensionless multiplier around 1.0. Most seasonal forcings
  (sinusoidal with `baseline = 1.0`), day-of-week reporting factors, school-term
  indicators.
- **`'per_day` / `'per_week` / `'per_month` / `'per_year`** — rate forcings. The
  value is in the named per-unit; the expander rescales it to the model
  `time_unit` (e.g. `'per_year` with `time_unit = 'days` multiplies all stored
  values by 1/365.2425).
- **`'count`** — raw population count. For `pop(t)` and similar demographic
  covariates.
- **`'days` / `'years`** — duration-valued forcings. Rare in practice.

---

## 8. Let Bindings

`let` declarations are **top-level** — they appear between blocks, never inside
a block. They are resolved after the full file is parsed (order does not matter
between `let` and other declarations).

```camdl
let N = S + E + I + R
let N_local[a in age, p in patch] = S[a,p] + E[a,p] + I[a,p] + R[a,p]
let foi[a in age, p in patch] = sum(b in age, C_age[a,b] * I[b,p] / N_local[b,p])
```

### 8.1 Scope Rules

**Bare names are always global.** `let N = S + E + I + R` means the total across
ALL strata. After age stratification, N is still the global total. No
auto-localization, ever.

**Indexed let bindings** define computed quantities over dimensions:

```camdl
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

A `sum` takes its binders first and its body last, and may bind **several axes
in one `sum`**. The flat form is sugar for the nested one — same value, same
IR:

```camdl
sum(a in age, p in patch, N[a, p])          # flat
sum(a in age, sum(p in patch, N[a, p]))     # the same reduction, nested
```

Each binder may carry its own `where` predicate (§8.2.1), and a later binder's
predicate may reference an earlier binder, exactly as in the nested form:

```camdl
sum(b in age where b != a, q in patch where dist[p,q] < 50, C[a,b] * I[b,q])
```

With no binder at all, `sum(e)` is just `e`. Since a bare family name already
means the total across all strata (§5.1), `sum(I)` is an explicit way to write
that total:

```camdl
sum(I)        # the total of I — exactly what bare `I` means
sum(S + I)    # total of S plus total of I; redundant, not wrong
```

For an indexed parameter, table or shaped `let`, the binder form is the
spelling — `sum(a in age, rho_a[a])`, not `sum(rho_a)`.

```camdl
# Define per-stratum totals
let N_local[a in age] = S[a] + E[a] + I[a] + R[a]

# N_local[child] = S[child] + E[child] + I[child] + R[child]
# N_local[adult] = S[adult] + E[adult] + I[adult] + R[adult]

# Sum them to get global total
# sum(a in age, N_local[a]) = N_local[child] + N_local[adult] = N
```

### 8.2.1 Restricted sums (`where`)

A `sum` may carry a `where` predicate that restricts the reduction to the index
values satisfying it:

```camdl
infection[p in patch] : S[p] --> I[p]
  @ beta * S[p] * sum(q in patch where dist[p,q] < 50 and q != p, I[q] / N[q])
```

The predicate is evaluated **at compile time**, so the sum expands to a term
only for the surviving index values. For a sparse spatial coupling this makes
the force-of-infection sum cost **O(P·k)** — roughly k neighbours per patch —
rather than O(P²), and by construction: it does not depend on any optimization
pass discovering that most weights are zero.

What a `where` predicate may reference is deliberately narrow, so the surviving
set is decidable before the simulation runs:

- **index variables**, compared with `==` / `!=` — `q != p`, `q == kano`;
- **constant table cells**, compared against a numeric literal with
  `< <= > >= == !=` — `dist[p,q] < 50`, `mask[p,q] != 0`. The table must be a
  compile-time constant (an inline literal or a `read(...)` table); the
  comparison is numeric (distance/mask tables are dimensionless — §2.1 has no
  length unit — so there is no dimensional ambiguity to check).

It may **not** reference a parameter, compartment state, or a parameterized
(non-constant) table cell — those are runtime quantities, and a
runtime-dependent support would change which terms exist as the simulation runs
(an unbounded reduction). A fitted-parameter threshold such as
`where dist[p,q] < sparse_thresh` is therefore rejected (E284): keep the
*support* a compile-time constant — a literal radius, or a precomputed 0/1 mask
table — and put any fitted **weight** in the rate body.

That separation is what lets a spatial kernel be both sparse and fittable. The
predicate carves the (fixed) support; the rate body holds the kernel, which may
carry parameters and so can be estimated:

```camdl
# constant `dist` carves the support; G and rho are fitted in the rate body
infection[p in patch] : S[p] --> I[p]
  @ beta * S[p] * ( I[p]/N[p]
      + G * sum(q in patch where dist[p,q] < 50 and q != p,
                dist[p,q]^(-rho) * I[q]/N[q]) )
```

Writing the *transition* per pair instead — `imp[p, q] : S[p] --> I[p] @ … where
p != q` — produces P² transitions (and as many flow columns) rather than one
summed rate per patch; the compiler warns (W105) and points back to the
`sum … where` form.

### 8.3 No Localization, No Magic

An unindexed global formula over a stratified compartment is a **hard error**,
not silently localized. There is no auto-transform that turns a global rate into
per-stratum indexed formulas: a bare mixing formula over stratified `S`/`I`
hard-errors (`E272` — "compartment is stratified but used without indices"), so
what runs is always what the user wrote.

The user writes the per-stratum force of infection explicitly:

```camdl
infection[a in age] : S[a] --> I[a]
  @ beta * S[a] * sum(b in age, C_age[a,b] * I[b] / N_local[b])
```

There is only one path to the indexed IR — the primitive above. The stratified
transmission surface (§10) is that same explicit indexed form, not a global
shorthand that expands behind the user's back.

### 8.4 Typed Let Bindings

A `let` binding may carry a type annotation from the same set of kinds used for
parameters: `rate`, `probability`, `positive`, `count`, `real`.

```camdl
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

```camdl
# Transfer: source --> destination
infection[a in age] : S[a] --> E[a]  @ beta * S[a] * I[a] / N_local[a]

# Inflow (exogenous): no source compartment
birth[p in patch] : --> S[child, p]
  @ mu * sum(a in age, N_local[a, p])

# Outflow: no destination
death_S[a in age, p in patch] : S[a,p] -->  @ mu_age[a] * S[a,p]

# Block form: rate on its own line (may also carry a `where` guard)
infection_water : S --> I {
  rate = S * beta_W * W / (K + W)
}
```

**Block form properties:**

- `rate` (required): the total propensity expression
- `where <guard>` (optional): an index-variable guard, identical to the inline
  `... @ <rate> where <guard>` form

**Inflows** (`-->` with nothing on the left) model individuals entering the
system from outside: births, importation, immigration. There is no source
compartment — the rate expression says how fast new individuals appear.
Stoichiometry: `[(destination, +1)]`.

**Importation** is an inflow of infected individuals from an external source,
typically at a constant or data-driven rate unrelated to the model's own state.
Unlike births (which depend on the existing population), importation represents
exogenous exposure — cases entering the modeled region from elsewhere:

<!-- camdl-doctest-preamble: importation
compartments { S, I, R }
parameters {
  beta        : rate
  gamma       : rate
  import_rate : rate
}
dimensions {
  age   = [child, adult]
  patch = [north, south]
}
tables {
  age_weights   : age   = [0.4, 0.6]
  patch_weights : patch = [0.5, 0.5]
}
stratify(by = age)
stratify(by = patch)
let N_local[a in age, p in patch] = S[a,p] + I[a,p] + R[a,p]
transitions {
  infection[a in age, p in patch] : S[a,p] --> I[a,p]
    @ beta * S[a,p] * I[a,p] / N_local[a,p]
  recovery[a in age, p in patch] : I[a,p] --> R[a,p] @ gamma * I[a,p]
}
-->

```camdl preamble=importation
transitions {
  # Constant importation rate (exogenous FOI)
  importation[a in age, p in patch] : --> I[a, p]
    @ import_rate * age_weights[a] * patch_weights[p]
}
```

### 9.1.1 Multi-source transitions (`A + B --> …`)

Either side of `-->` accepts a `+`-separated list of compartments. Each source
contributes `-1` and each destination contributes `+1` to the transition's net
stoichiometry. This exposes the IR's already-general
`stoichiometry: Vec<(name, int)>` shape to the DSL — the same construct the Rust
runtime has always consumed.

**Bimolecular mass-action** (vector-host transmission, pair formation, cell-cell
interactions):

```camdl
# Host infection from an infectious mosquito bite.
bite     : S_h + I_v --> I_h + I_v   @ a * b_h * S_h * I_v / H

# Mosquito infection from biting an infectious host.
infect_v : S_v + I_h --> E_v + I_h   @ a * b_v * S_v * I_h / H

# Chemistry-style two-reactant reaction.
react    : A + B --> C               @ k * A * B
```

Atomic firing: a single Gillespie step applies the _vector_ of deltas at once.
For `bite`, `S_h` decrements and `I_h` increments together — no intermediate
state.

**Backend requirement.** A multi-source transition requires **Gillespie or
ODE**. Chain-binomial rejects it with a hard error (gh#121): its competing-risk
draw bounds the drawn flow by a *single* source count, which is wrong when two
sources jointly gate the event (the correct joint draw, bounded by the minimum
over all sources, is not implemented). On chain-binomial, fall back to the
single-source encoding with the second population referenced only in the rate
(shown under "When to use it" below).

**Catalyst collapse.** A compartment appearing on both sides of the arrow
contributes `−1` and `+1`, which sum to `0`. Zero-delta entries are dropped from
the stoichiometry because (a) they produce no net state change, and (b) the IR
validator rejects `delta == 0`. The rate expression's reference to the catalyst
is preserved, so the propensity dependency graph still wires it in.

- `bite : S_h + I_v --> I_h + I_v @ …` → stoichiometry `{S_h: -1, I_h: +1}`.
- `react : A + B --> C @ …` → stoichiometry `{A: -1, B: -1, C: +1}`.

**No-net-effect error (E310).** A transition where every delta collapses to zero
(all compartments are pure catalysts) emits `E310`: "transition 'X' has no net
effect: sources and destinations cancel". Almost always a model error; the hint
suggests removing catalysts or adding a non-trivial destination.

**When to use it.** Multi-source is the canonical encoding for any
two-population mass-action event — transmission that depends on two pop classes
(S × I), predator-prey-style interactions, chemistry. Single-source with the
catalyst referenced only in the rate is equivalent in the IR but hides the
biological structure:

```camdl
# Clear (reads like the biology).
bite : S_h + I_v --> I_h + I_v  @ a * b_h * S_h * I_v / H

# Equivalent IR, less clear.
bite : S_h --> I_h              @ a * b_h * S_h * I_v / H
```

Both produce the same stoichiometry `{S_h:-1, I_h:+1}` and the same rate
expression. Prefer the multi-source form when two compartments jointly determine
the event — the `+` makes the dependency explicit.

### 9.1.2 Probabilistic branching (`X --> { A : p, B : 1-p }`)

When a single event has multiple possible destination compartments chosen
probabilistically — symptomatic vs asymptomatic infection, mild vs severe vs
fatal progression, detected vs missed case — write the destinations as a
weighted set:

```camdl
# An infection event produces a symptomatic case with probability
# p_symp, otherwise asymptomatic.
infection : S --> { I_symp : p_symp, I_asym : 1 - p_symp }
  @ beta * S * (I_symp + I_asym) / N

# Age-indexed branching (malaria use case).
bite[a in age] : X[a] --> { Y_symp[a] : p_symp[a], Y_asym[a] : 1 - p_symp[a] }
  @ h_eff * X[a]
```

**Semantics.** Pure compile-time sugar. Each branch expands to its own IR
transition with rate `weight_i × original_rate` and stoichiometry
`{source: -1, dest_i: +1}`. The existing source- grouping machinery in every
stochastic backend (Gillespie, chain-binomial) correctly groups
transitions sharing a source and performs a **single multinomial split** at
firing time — _not_ two independent draws, which would double-consume the source
at high incidence. See `sim/src/chain_binomial.rs` §Euler-multinomial for the
algorithm (matches pomp's `reulermultinom`).

For the example above, `S --> { I_symp : p_symp, I_asym : 1 - p_symp }
@ r`
produces two IR transitions:

| Name               | Stoich              | Rate               |
| ------------------ | ------------------- | ------------------ |
| `infection_I_symp` | `{S:-1, I_symp:+1}` | `p_symp * r`       |
| `infection_I_asym` | `{S:-1, I_asym:+1}` | `(1 - p_symp) * r` |

The total exit rate from `S` is `p_symp * r + (1 - p_symp) * r = r`, so the
overall depletion rate is unchanged; only the destination distribution is
refined.

**Branch naming.** The compiler appends the destination compartment name to the
transition's base name to disambiguate. Indexed transitions combine the index
suffix and branch suffix:
`bite[a in age] : X[a] --> { Y_symp[a] : …, Y_asym[a] : … } @ …` produces
`bite_child_Y_symp`, `bite_child_Y_asym`, `bite_adult_Y_symp`,
`bite_adult_Y_asym`.

**When to use it.** When the biology is "one event, multiple outcomes chosen
stochastically at the moment of the event." When the biology is "two separate
ongoing processes from the same compartment with different rates" (e.g., death
and recovery from I), use two plain transitions — the runtime treats them
identically under source-grouping, but the DSL intent differs and two
transitions reads more naturally.

**Weights.** The weight of each branch is any scalar expression with dimension
`probability` (dimensionless, domain `[0, 1]`). The compiler does not enforce
that weights sum to 1 — users can write rate-weighted branches where the sum
differs from 1 (e.g., for a fraction of events going to an "other" compartment
that's implicit). Most users will write `{A : p, B : 1 - p}` for binary
branching or an explicit last entry `1 - sum-of-others` for n-way branching.

### 9.2 Indexed Transitions

```camdl
transition_name[i in dim1, j in dim2, ...] : from --> to  @ rate
```

The `[i in dim]` clause binds index variables. The compiler generates one
concrete IR transition per combination of index values. Dimensionality is known
at compile time: `|dim1| × |dim2| × ...` transitions.

### 9.3 Guard Clauses (`where`)

The `where` clause filters which index combinations generate transitions:

```camdl
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

<!-- camdl-doctest-preamble: consecutive-aging
compartments { S, I, R }
dimensions {
  age   = [a0_5, a5_15, a15_50, a50_65, a65p]
  patch = [north, south]
}
tables { age_dur : age = [5.0, 10.0, 35.0, 15.0, 20.0] }
stratify(by = age)
stratify(by = patch)
-->

```camdl preamble=consecutive-aging
transitions {
  aging[c in compartments, (a, a_next) in consecutive(age), p in patch]
    : c[a, p] --> c[a_next, p]
    @ (1 / age_dur[a]) * c[a, p]
}
```

For `age = [age_0_5, age_5_15, age_15_50, age_50_65, age_65p]`, this generates
four transitions per compartment per patch: `age_0_5→age_5_15`,
`age_5_15→age_15_50`, `age_15_50→age_50_65`, `age_50_65→age_65p`. The last
stratum has no outgoing aging transition.

This is a general-purpose primitive for any sequential transfer along an ordered
dimension. It also handles **Erlang sub-staging** for non-exponential waiting
times:

<!-- camdl-doctest-preamble: erlang-E
compartments { E, I }
parameters { sigma : rate }
-->

```camdl preamble=erlang-E
# Erlang-3 latent period: E passes through 3 sub-stages
dimensions { erlang_E = [e1, e2, e3] }
stratify(by = erlang_E, only = [E])

transitions {
  progression[(s, s_next) in consecutive(erlang_E)]
    : E[s] --> E[s_next]
    @ 3 * sigma * E[s]       # k * sigma for Erlang-k

  # Final sub-stage transitions to I
  progression_final : E[e3] --> I
    @ 3 * sigma * E[e3]
}
```

This gives an Erlang(k=3, rate=sigma) distributed latent period. The mean is the
same as exponential (1/sigma), but the variance is reduced by factor k,
producing a more peaked distribution — closer to real disease progression.

Writing the chain by hand works, but it buries the modelling intent and invites
a silent error: the per-stage rate must be `k * sigma`, and writing plain
`sigma` gives a mean wrong by a factor of `k` with no diagnostic. The `via`
clause below states the residence directly and lowers to exactly this chain.

#### 9.4.1 Staged residences: the `via` clause

A compartment's residence time is exponential by default: while an individual is
in `E`, it leaves at the total exit hazard — memoryless, with coefficient of
variation (CV, the standard deviation over the mean) equal to one. Real latent
and infectious periods are more regular than that, and the number of stages in a
period materially changes epidemic speed and the reproduction number inferred
from the same data. The **method of stages** recovers a non-exponential dwell by
splitting a compartment into `k` internal sub-stages, each exponential of rate
`k/τ`, so the total residence is Erlang(`k`, `k/τ`): mean `τ`, variance `τ²/k`,
CV `1/√k`. The manual `consecutive` staging above is exactly this construction,
written out.

A transition that **drains** a compartment can carry that residence law
directly. In place of a rate it takes a `via` clause naming the law, which
stages the **source** compartment and supplies the per-stage rate for you:

<!-- camdl-doctest-preamble: via-erlang
compartments { S, E, I, R }
parameters {
  beta  : rate
  sigma : rate
  gamma : rate
}
-->

```camdl preamble=via-erlang
transitions {
  infection : S --> E  @ beta * S * I / (S + E + I + R)  # the force of infection (ordinary event)
  onset     : E --> I  via erlang(stages = 3, rate = sigma)   # E's residence is Erlang-3
  recovery  : I --> R  via erlang(stages = 3, rate = gamma)   # I's residence is Erlang-3
}
```

A transition is **either** `@ rate` **or** `via law`, never both and never
neither. `@ rate` is an ordinary exponential transition — the rate is the
propensity; `via law` is a staged residence — the law supplies `k/τ` per stage.
The force that *fills* the compartment (here `infection`) stays an ordinary
`@`-transition; only the draining transition carries the dwell law, so there is
no "entry force versus residence rate" ambiguity.

`via erlang(...)` lowers to ordinary sub-staged compartments and `consecutive`
transitions — `E` becomes `E_s1, E_s2, E_s3`, chained at rate `3 * sigma` and
exiting into `I_s1` — isomorphic (modulo stage names) to the hand-written form
in §9.4. Because the sub-stages are ordinary compartments, a **bare** reference
to the staged source sums over its stages automatically (the bare-name rule,
§5.1): the `I` in the force of infection above already means
`I_s1 + I_s2 + I_s3`, so the mean infectious period and R₀ are preserved and only
the dwell-time *shape* changes. `via` introduces no new IR — it is a macro over
existing compartments and transitions, so all three backends and the gradient
machinery see the lowered model unchanged.

In `erlang(stages = k, mean = τ | rate = r)`, give exactly one of `mean` or
`rate` (they are reciprocals); `stages` is a positive-integer literal, because it
sets how many compartments exist — model structure, not a fittable parameter.
`mean`, `rate`, and mixture weights *are* fittable and gradient-estimable.
`stages = 1` is the ordinary exponential dwell — not a no-op, but the `k`-knob at
its lowest setting.

**Mixtures of durations.** Some periods are not a single Erlang but a mixture:
an individual takes one of several routes, chosen on entry. `hyper_erlang` is a
finite mixture of Erlang chains, each written as a self-contained `branch(...)`.
When the branches share an endpoint, write it once on the arrow and give each
branch a `weight`; the last branch's weight is implicit (`1 −` the sum of the
others), so the mixture is normalized by construction:

<!-- camdl-doctest-preamble: via-hyper-shared
time_unit = 'weeks
compartments { S, I, R }
parameters {
  beta    : rate
  p       : probability
  tau_typ : positive
  tau_pro : positive
}
-->

```camdl preamble=via-hyper-shared
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  clearance : I --> R via hyper_erlang(
    branch(label = typical,   weight = p, stages = 2, mean = tau_typ),
    branch(label = prolonged,             stages = 1, mean = tau_pro)
  )
}
```

When the branches end in *different* compartments — a case-fatality split, where
the fatal and recovering arms have different durations *and* destinations — each
branch carries its own `to` and the transition needs no arrow target (a branch
with neither a `to` nor a transition target is an error):

<!-- camdl-doctest-preamble: via-hyper-dest
time_unit = 'days
compartments { S, E, I, R, D }
parameters {
  beta  : rate
  sigma : rate
  cfr   : probability
}
-->

```camdl preamble=via-hyper-dest
transitions {
  infection : S --> E @ beta * S * I / (S + E + I + R)
  onset     : E --> I @ sigma * E
  outcome   : I via hyper_erlang(
    branch(label = fatal,   weight = cfr, stages = 3, mean =  8 'days, to = D),
    branch(label = recover,               stages = 3, mean = 12 'days, to = R)
  )
}
```

Each branch is `branch(label, stages, mean | rate, weight?, to?)`: `label` is a
required bare name (it names the branch's stage compartments, `I__fatal__1 …`),
and `weight` and `to` are optional. This is *not* two competing exponential
exits — those would give coupled, exponential outcome times; here the outcome is
decided on entry and each arm runs its own gamma chain to its own destination.
The bare `I` in the force of infection still sums *all* the branch stages
(everyone infectious), so transmission is unaffected.

The clause also has a **block form**, mirroring the brace body used elsewhere,
with `rate` and `via` mutually exclusive:
`onset : E --> I { via = erlang(stages = 3, rate = sigma) }`.

**Scope and diagnostics.** A staged compartment must be drained by exactly one
`via` transition; a second draining exit — another `via`, or an ordinary `@`
racing with the dwell — is the *competing-exit* case, rejected with `E246` and
left to the manual per-stage form for now. The laws that ship are `erlang` and
`hyper_erlang`; any other law (`coxian`, `fixed`, …) is `E243`. `hyper_erlang`
on an already-stratified source is a later sub-phase, rejected today with `E248`.
Argument mistakes each get a named code: a `stages` that is not a positive
integer is `E244`; giving both or neither of `mean`/`rate` is `E245`; an unknown
`erlang` keyword is `E247`; fewer than two branches is `E255`; a `weight` on the
last branch (or a missing one on any earlier branch) is `E256`; and duplicate
branch labels are `E258`.

**Entering and targeting a staged compartment.** Filling a staged compartment is
asymmetric with reading it. `init` and inflow transitions (`--> E`) land in
**stage 1** automatically — write the bare name and the compiler routes the
arrival to `E_s1`. An intervention is stricter: `transfer(to = E)` naming the
bare staged compartment is **`E237`**, because the stage axis is synthesized per
`via` transition and is private to it, so it never pairs cell-for-cell with the
other endpoint's shape — name the explicit stage (`transfer(to = E_s1, …)`)
instead, which is what the diagnostic's hint suggests. Reads are unaffected: a
bare `E` or `prevalence(E)` still sums every stage (the bare-name rule, §5.1).

#### 9.4.2 Aging across a stratified model (canonical use case)

`consecutive(dim)` is also the right primitive for **demographic aging across
age bins** in any stratified model. Combined with `c in
compartments` and an
outer `[s in setting]` binding, one declaration covers all compartment families
and all outer strata:

```camdl
dimensions {
  setting = [low_burden, mid_burden, high_burden]
  age     = [a02, a25, a510, a1015, a15plus]
}

compartments { S, I, R }
stratify(by = setting)
stratify(by = age)

tables {
  age_dur : age = [2.0, 3.0, 5.0, 5.0, 0.0]   # years; last is open-ended
}

transitions {
  aging[c in compartments, s in setting, (a, a_next) in consecutive(age)]
    : c[s, a] --> c[s, a_next]
    @ (1 / (age_dur[a] * 365.0)) * c[s, a]
}
```

For 3 settings × 3 compartments × 4 age boundaries this expands to 36 IR
transitions — from one DSL declaration. Without `consecutive(dim)` the same
pattern needs 36 hand-written lines (one per compartment/setting/boundary), all
structurally identical, and a new boundary requires editing every compartment
family.

If you find yourself writing `age_S_02`, `age_S_25`, `age_S_5`, ...
hand-enumerated transitions, you want this primitive.

### 9.5 Compartment Iteration

The `c in compartments` binding iterates over compartment names:

<!-- camdl-doctest-preamble: compartment-iter
compartments { S, I, R }
parameters { mu : rate }
dimensions {
  age   = [child, adult]
  patch = [north, south]
}
tables {
  mig : patch × patch = [[0.0, 0.1],
                         [0.1, 0.0]]
}
stratify(by = age)
stratify(by = patch)
-->

```camdl preamble=compartment-iter
transitions {
  # Death for all compartments
  death[c in compartments, a in age, p in patch] : c[a,p] -->
    @ mu * c[a,p]

  # Migration for all compartments
  migrate[c in compartments, a in age, src in patch, dst in patch]
    : c[a,src] --> c[a,dst]
    @ mig[dst,src] * c[a,src]
    where src != dst
}
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
cannot be chained: `a < b < c` is a parse error. In a `where` predicate,
relational operators (`< <= > >= == !=`) compare a constant table cell to a
literal (`dist[p,q] < 50`); index variables support only `==` / `!=` (see
§8.2.1).

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

In a `where` guard (transition or `sum` predicate) the comparison operators are
restricted — see §8.2.1: index variables compare only with `==` / `!=`, and the
relational operators (`< <= > >= == !=`) apply only to a constant table cell
against a numeric literal (`dist[p,q] < 50`). `sum` is a keyword, not a
user-definable function.

**Built-in math functions.** These are recognized by the compiler as
function-call syntax and produce IR expression nodes (not forcing functions):

| Function    | Arity | Result                                                          |
| ----------- | ----- | --------------------------------------------------------------- |
| `exp(x)`    | 1     | e^x                                                             |
| `log(x)`    | 1     | Natural logarithm (ln). Returns -∞ for x ≤ 0.                   |
| `sqrt(x)`   | 1     | Square root. Returns 0 for x < 0.                               |
| `abs(x)`    | 1     | Absolute value                                                  |
| `floor(x)`  | 1     | Floor (round toward -∞)                                         |
| `ceil(x)`   | 1     | Ceiling (round toward +∞)                                       |
| `sin(x)`    | 1     | Sine (radians)                                                  |
| `cos(x)`    | 1     | Cosine (radians)                                                |
| `tanh(x)`   | 1     | Hyperbolic tangent                                             |
| `mod(a, b)` | 2     | Euclidean remainder (always non-negative). Returns 0 for b = 0. |
| `min(a, b)` | 2     | Smaller of two values                                          |
| `max(a, b)` | 2     | Larger of two values                                           |

Example:

```camdl
let day_of_year = mod(t, 365.25)
let pop_decay = N0 * exp(-mu * t)
let is_pulse = (day_of_year > 250.0) * (day_of_year < 252.0)
```

**Rate wrappers.** Two compiler-recognized forms modify how event counts are
drawn for a transition. They are NOT general-purpose functions — they wrap the
entire rate expression and are extracted by the compiler during expansion.

| Wrapper                   | Syntax                                        | Effect                                                       |
| ------------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| `overdispersed(rate, σ²)` | `@ overdispersed(beta * S * I / N, sigma_se)` | Gamma-Poisson (NegBinomial) draws. Var = mean + mean²·σ²/dt. |
| `deterministic(rate)`     | `@ deterministic(mu * N)`                     | Rounded integer: nearbyint(rate × dt). No stochastic noise.  |

These are documented in §9.8 (overdispersion) and are compatible with the
chain-binomial backend. Gillespie and ODE reject models with
`overdispersed()` transitions.

`deterministic(rate)` fires an exact count of `nearbyint(rate · dt)`, clamped to
`[0, n_src]` — it never removes more than the source population. It is supported
**only as the sole exit** from its source: a source that has a
`deterministic(...)` exit *and* any other competing exit (a stochastic exit, or
even a second `deterministic(...)`) is rejected (gh#122), because the flows would
be drawn independently and could together over-draw the source. (The ODE backend
runs *every* transition deterministically, so the restriction does not apply
there.)

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
namespaces. User names cannot shadow reserved identifiers (see §14).

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
second is σ²_SE, the **intensity** of the Gamma white noise — the variance the
underlying Gamma process accumulates per unit time (He et al. 2010). It is *not*
the realized multiplier variance: over a substep of length `dt` the runtime
draws a mean-one multiplicative factor `G ~ Gamma(shape = dt/σ², scale = σ²/dt)`,
so `E[G] = 1` and `Var(G) = σ²/dt` — the realized variance scales inversely with
the step. The resulting event count distribution is NegBinomial — the
Poisson-Gamma compound.

`overdispersed` is syntactically a function call in expression position. The
compiler extracts it during expansion: the inner rate goes to `transition.rate`,
the variance goes to `transition.overdispersion` in the IR. Transitions without
`overdispersed` have `overdispersion: null` — standard Poisson draws.

**Backend compatibility.** Overdispersion is incompatible with Gillespie SSA
(which assumes deterministic rates between events) and meaningless for ODE
(deterministic). It is supported by chain-binomial (NegBinomial replaces Poisson
draws). The runtime enforces this: requesting `--backend gillespie` for a model
with `overdispersed` transitions produces a hard error with a hint to use
`--backend chain_binomial`.

**Composability.** Each transition independently chooses whether to be
overdispersed, and with what variance:

```
infection : S --> I  @ overdispersed(beta * S * I / N, sigma_inf)
recovery  : I --> R  @ overdispersed(gamma * I, sigma_rec)
waning    : R --> S  @ omega * R   # no extra noise
```

---

## 10. Stratified Transmission (Explicit Indexed Form)

> **The `coupling[dim = M]` sugar was removed.** An earlier design had a
> `coupling[dim = M]` block that auto-expanded a base transmission rate into a
> contact-matrix-weighted sum; it was tried and removed, and there is no
> `coupling` keyword in the grammar. Write stratified transmission with the
> explicit indexed transition form below. For **sparse spatial coupling** the
> recommended construct is a restricted sum, `sum(q in dim where P, body)`
> (§8.2.1), which carves the neighbour support at compile time so the
> force-of-infection sum costs O(P·k) rather than O(P²).

### 10.1 The Explicit Primitive

Write the full indexed transmission formula directly — it is always correct and
always available. For models with multiple stratification dimensions the formula
is longer but fully transparent:

```camdl
# Fully explicit age × sex structured transmission
infection[a in age, s in sex] : S[a,s] --> E[a,s]
  @ beta * S[a,s] * sum(b in age, sum(t in sex,
      C_age[a,b] * B_sex[s,t] * I[b,t]
        / sum(c in compartments, c[b,t])
    ))
```

The per-stratum denominator `sum(c in compartments, c[b,t])` is the total
population of stratum `(age=b, sex=t)` across all compartments; you can also
declare it once as a `let N_local[...] = ...` binding (§8.4) and reference it.

### 10.2 What the Matrices Mean

All coupling structures are expressed through the same mechanism — a rate matrix
`M[i,j]` weighting contact between strata i and j:

| Matrix structure  | Effect                                  | Example             |
| ----------------- | --------------------------------------- | ------------------- |
| Dense             | General mixing                          | Age contact matrix  |
| Off-diagonal only | Directed (no within-group transmission) | STI sex-structured  |
| Identity          | Within-stratum only                     | Same as no coupling |
| All ones          | Homogeneous mixing                      | No structure        |

<!-- camdl-doctest-preamble: table-matrices
compartments { S, I, R }
dimensions {
  age = [child, adult]
  sex = [female, male]
}
stratify(by = age)
stratify(by = sex)
parameters {
  beta_mf : rate
  beta_fm : rate
  gamma   : rate
}
let N_local[a in age, s in sex] = S[a,s] + I[a,s] + R[a,s]
transitions {
  infection[a in age, s in sex] : S[a,s] --> I[a,s]
    @ S[a,s] * sum(b in age, sum(t in sex, C_age[a,b] * B_sex[s,t] * I[b,t] / N_local[b,t]))
  recovery[a in age, s in sex] : I[a,s] --> R[a,s] @ gamma * I[a,s]
}
-->

```camdl preamble=table-matrices
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

### 10.3 Multi-Strain Models

Multi-strain models use the explicit indexed transition form throughout.

The key structural insight: in a multi-strain compartmental model, **S is a
shared pool** — a susceptible person isn't "susceptible to wild-type," they're
just susceptible. The strain dimension belongs on E, I, R (tracking which strain
you're infected with / recovered from), not on S.

```camdl
compartments { S, E, I, R }

dimensions {
  age    = [child, adult]
  strain = [wt, delta]
}

stratify(by = age)
stratify(by = strain, only = [E, I, R])

parameters {
  beta  : rate
  sigma : rate
  gamma : rate
}

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

The `ode { }` block declares derivatives for real-valued compartments
(`W : real`). The expander emits each `W = dW/dt` line as an IR ODE equation, and
the runtime integrates them (RK4) between stochastic events — a
piecewise-deterministic Markov process. A `Real` compartment must have an ODE
equation; one without is a compile error.

For real-valued compartments:

<!-- camdl-doctest-preamble: ode-demo
compartments { S, I, R, W : real }
parameters {
  beta  : rate
  gamma : rate
  xi    : rate
  delta : rate
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
-->

```camdl preamble=ode-demo
ode {
  W = xi * I - delta * W      # dW/dt = xi * I - delta * W
}
```

Left side = compartment name, right side = time derivative. Creates a
piecewise-deterministic Markov process (PDMP): stochastic events for integer
compartments, ODE evolution for real compartments between events.

**Interaction with transitions, effects, and balance (ODE backend).** When the
whole model is run on the fully-deterministic ODE backend, *every* transition
also feeds the derivatives: each adds `stoichiometry · rate` to its
compartments' `dc/dt` (integer compartments evolve as deterministic flow, not by
stochastic draws), and each `real` compartment's `dW/dt` comes from its `ode {}`
equation. Scheduled effects (interventions and events) are applied as **exact
discontinuities** — the integrator lands exactly on the effect time, applies the
effect, records output post-effect (§13.10), and restarts from the modified
state. A `balance {}` constraint is **not** available on ODE: balance is a
chain-binomial-only capability, so a model carrying a `balance {}` block run on
ODE fails at dispatch with a capability error naming the limitation.

---

## 12. Observations

<!-- camdl-doctest-preamble: obs-sir
compartments { S, I, R }
parameters {
  beta     : rate
  gamma    : rate
  rho      : probability
  k        : positive
  p_detect : probability
  N        : count
  N_tested : count
  rho_sens : probability
  rho_spec : probability
}
let Ntot = S + I + R
transitions {
  infection : S --> I @ beta * S * I / Ntot
  recovery  : I --> R @ gamma * I
}
-->

```camdl preamble=obs-sir
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }

  detection {
    columns       { time : time, detection : count }
    projected     = prevalence(I)
    emit_schedule = every 14 'days
    detection     ~ bernoulli(p = p_detect * projected / N)
  }
}
```

Syntax notes: the observation header is `name { … }` (no colon), optionally
`name from <source> { … }`. `columns { }` declares the file schema by name
(exactly one `: time` column). The measurement model uses `<value_col> ~
KIND(...)` (the `~` operator, function-call form with named arguments).
`emit_schedule` is the simulate-only emission cadence; a fit-only model omits
it (the data file's `time` column drives the fit).

### 12.1 Projections

```camdl
incidence(transition)                    cumulative flow since last observation
incidence(transition[north])             positional index, by declaration order
incidence(transition[patch = north])     named index (order-independent)
prevalence(compartment)                  current population
prevalence(compartment[age = child])     named index on compartment
```

An index on a projection **selects one cell of a stratified family; it never
marginalizes.** Every declared dimension must be given, positionally or by name
(the two forms may be mixed). Dropping a dimension is an error, not a silent sum
— see "A partial index does not marginalize" below for the explicit forms to
write instead. On a `[patch, age]` family, `infection[patch = north]` names no
cell; `infection[patch = north, age = child]` names `infection_north_child`.

`incidence(...)` takes **exactly one** transition. Several arguments is `E203`
and none is `E250`; `incidence(a, b)` does not sum, and neither does
`incidence(a) + incidence(b)`, because these heads are sugar rather than
expression functions (see "head-position sugar" below). To observe two flows
as one column, pool strata of a single family with
`sum(a in dim, incidence(tr[a]))`, or route distinct flows through one junction
transition. `prevalence(...)` differs: several arguments desugar to their sum
(`prevalence(X1, X2)` is `prevalence(X1 + X2)`), because a compartment
population is an expression leaf and a flow is not.

**Arithmetic projections** (the general form). Beyond `incidence()` and
`prevalence()` sugar, `projected` accepts any expression over compartment state,
parameters, and time. Pooled-group prevalence, prevalence-as-proportion, and
arbitrary derived observables compose naturally:

<!-- camdl-doctest-preamble: obs-derived
compartments { S, I_m, I_s, R, x3, y3 }
parameters {
  beta     : rate
  gamma    : rate
  N_tested : count
  rho_sens : probability
  rho_spec : probability
}
let Ntot = S + I_m + I_s + R
transitions {
  infection : S --> I_m @ beta * S * (I_m + I_s) / Ntot
  recovery  : I_m --> R @ gamma * I_m
}
-->

```camdl preamble=obs-derived
observations {
  # Pooled-group count (Garki patent prevalence across x3, y3).
  patent_count {
    columns       { time : time, patent_count : count }
    projected     = x3 + y3
    patent_count  ~ poisson(rate = projected)
    emit_schedule = every 1 'months
  }

  # Prevalence-as-proportion — the canonical surveillance form.
  slide_positivity {
    columns          { time : time, slide_positivity : count }
    projected        = (I_m + I_s) / (S + I_m + I_s + R)
    slide_positivity ~ diagnostic_test(
      base = binomial(n = N_tested, p = projected),
      sens = rho_sens, spec = rho_spec
    )
    emit_schedule    = every 1 'weeks
  }
}
```

Arithmetic projections emit `Ir::Projection::DerivedExpr`. Both the
forward-simulation emission path (`camdl simulate --obs`) and the
likelihood-scoring path (pfilter / PGAS / IF2) share a single evaluator
(`sim::inference::multi_stream_obs::eval_stream_projection`), so they agree on
semantics by construction. If you find yourself wanting a "multi-compartment
prevalence" shortcut, write the sum directly: `projected = x + y` is the general
form; `prevalence(x)` is kept as sugar only for the single-compartment case
where the named function clarifies intent.

**A partial index does not marginalize; state cross-strata aggregation
explicitly.** Dropping *some* of a stratified family's dimensions is an error,
not a silent sum: `incidence(infection[patch = north])` over a `[patch, age]`
family resolves to a single expanded transition and fails if none matches
(there is no `infection_north` — only `infection_north_child`, …). To sum a
projection across a dimension, write the sum out:

`incidence(...)` and `prevalence(...)` are **head-position sugar**: they are
valid as the whole of a `projected = …` right-hand side, or under `sum(...)`,
but they are not expression functions and cannot be wrapped in arithmetic
(`rho * sum(p in patch, incidence(infection[p]))` is `E100`, undeclared
function). The reporting rate therefore goes in the *likelihood*, not the
projection:

- one pooled column, one reporting rate:

  ```camdl
  projected = sum(p in patch, incidence(infection[p]))
  cases ~ poisson(rate = rho * projected)
  ```

  A family stratified over several dimensions needs one `sum` per dimension:
  `sum(a in age, sum(p in patch, incidence(infection[a, p])))`.

- one row per stratum, each with its own reporting rate — index the stream:

  ```camdl
  cases[p in patch] {
    columns   { time : time, patch : dim, cases : count }
    projected = incidence(infection[p])
    cases ~ poisson(rate = rho[p] * projected)
  }
  ```

A `where` predicate on an aggregation sum prunes the domain, exactly as it does
in a rate expression: `sum(p in patch where p != north, incidence(infection[p]))`
pools only the surviving levels.

A **bare, un-indexed** `incidence(infection)` over a stratified family on an
un-indexed observation stream is rejected (`E280`) precisely so this aggregation
decision is never made silently; the diagnostic prints the explicit forms
above, naming the family's actual dimensions. Where you *do* fully index, prefer **named** indexing
(`infection[patch = north, age = child]`) over **positional**
(`infection[north, child]`): named binding is order-independent and survives a
later reordering of the dimension declarations, whereas positional binding
silently re-interprets against the new order.

Named indices resolve by the dimension they name, in any order (gh#459), and a
label that is not a dimension of the family is `E332`. Positional indices carry
no such protection: they are matched to slots by position and checked only
against that slot's levels, so when two dimensions share level names a swapped
pair names a *different existing cell* and binds silently, with no diagnostic
(§1).

Inside a likelihood expression, the keyword `projected` refers to the evaluated
projection value for that observation.

### 12.2 Likelihood Families

```camdl
neg_binomial(mean = EXPR, r = EXPR)            overdispersed counts
poisson(rate = EXPR)                           Poisson counts
normal(mean = EXPR, sd = EXPR)                 continuous
binomial(n = EXPR, p = EXPR)                   bounded counts
beta_binomial(n = EXPR, alpha = EXPR, beta = EXPR)          overdispersed prevalence (raw)
beta_binomial(n = EXPR, mean = EXPR, concentration = EXPR)  overdispersed prevalence (mean/concentration)
beta(mean = EXPR, concentration = EXPR)        continuous proportion in (0, 1)
bernoulli(p = EXPR)                            binary outcome
```

`neg_binomial(mean = μ, r = k)` is the **NB2** (mean–dispersion)
parameterization: the mean is `μ` and the variance is `μ + μ²/r`, so a smaller
`r` means more overdispersion and `r → ∞` recovers `poisson(μ)`. A prior on `r`
is therefore a prior on the quadratic excess variance — pin it deliberately.

**Count arguments carry the count dimension `[P]`.** `neg_binomial`'s `mean`,
`poisson`'s `rate`, and `binomial`/`beta_binomial`'s `n` are all *expected
counts over the reporting interval*, not per-time rates. Writing a transition
rate there (`projected = gamma * I`, dimension `P·T⁻¹`) is **E304**. The
distinction is invisible at a one-day reporting step, where a per-day rate and a
one-day accumulated count coincide numerically; at a weekly step the likelihood
is wrong by roughly the window length. Use `incidence(<transition>)`, which
accumulates the flow over the interval. A bare numeric literal (`mean = 100`) is
a count by context and is exempt.

The two `beta_binomial` spellings are equivalent: `mean`/`concentration` lowers to
`alpha = mean · concentration`, `beta = (1 − mean) · concentration`. Use whichever
reads better; mixing the two forms in one call is an error (E252). For a
sampler-friendly parameterization of the `concentration` (overdispersion) parameter,
see the reparameterization guidance in `camdl docs inference`.

Use `beta(...)` when the observed value is itself a **continuous proportion** in the
open interval (0, 1) — a rate, coverage, or positivity given directly as a fraction —
rather than a `k`-of-`n` count (which is `beta_binomial`). It is mean-linked with the
same shape mapping (`alpha = mean · concentration`, `beta = (1 − mean) · concentration`);
`mean` and `concentration` are both differentiable, so `beta` is usable under
gradient-based inference (`nuts`) as well as the gradient-free methods.

### 12.2.1 Diagnostic-test likelihood sugar

Surveillance data is almost never perfectly observed — slide microscopy, RDTs,
and PCR all have sensitivity < 1 and specificity < 1. The `diagnostic_test`
sugar absorbs the measurement-model correction so the DSL reads like the
biology:

```camdl preamble=obs-sir
observations {
  slide_positivity {
    columns          { time : time, slide_positivity : count }
    projected        = prevalence(I)
    emit_schedule    = every 1 'weeks
    slide_positivity ~ diagnostic_test(
      base = binomial(n = N_tested, p = projected / N),
      sens = rho_sens,
      spec = rho_spec
    )
  }
}
```

**Semantics.** Pure compile-time rewrite. If true positive fraction is π, the
probability of a positive test outcome is

```
p_observed  =  sens · π  +  (1 − spec) · (1 − π)
```

— the first term is a true-positive (infected and detected), the second is a
false-positive (uninfected but mistakenly positive). The compiler rewrites the
inner likelihood's `p = π` to this expression, producing IR byte-identical to a
hand-inlined

```camdl
likelihood = binomial(
  n = N_tested,
  p = rho_sens * projected / N + (1 - rho_spec) * (1 - projected / N)
)
```

**Supported bases.**

- `binomial(n = …, p = …)` — survey of `n` individuals, count positives.
- `bernoulli(p = …)` — single test per individual, 0/1 outcome.

Other likelihood families (`poisson`, `neg_binomial`, `normal`, `beta_binomial`)
aren't meaningful as diagnostic-test bases (sens/spec correct a probability, not
a count-mean or variance) and produce `E253` when used.

**Parameters.** `sens` and `spec` can be anything in `[0, 1]`: fixed constants,
parameters with priors (for joint estimation of test characteristics with the
transmission model), or expressions. Dimensional type is `probability`; the
compiler checks domain.

**Diagnostics.**

- `E253` — base must be `binomial(...)` or `bernoulli(...)`; other likelihood
  families rejected.
- `E254` — missing one of the required keyword arguments `base`, `sens`, `spec`.

### 12.3 Indexed Observations

<!-- camdl-doctest-preamble: obs-patch
compartments { S, I, R }
dimensions {
  patch = [north, south, east]
}
stratify(by = patch)
parameters {
  beta  : rate
  gamma : rate
  rho   : probability
  k     : positive
}
let N[p in patch] = S[p] + I[p] + R[p]
transitions {
  infection[p in patch] : S[p] --> I[p] @ beta * S[p] * I[p] / N[p]
  recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]
}
-->

```camdl preamble=obs-patch
observations {
  cases_by_patch[p in patch] {
    columns        { time : time, patch : dim, cases_by_patch : count }
    projected      = incidence(infection[patch = p])
    emit_schedule  = every 7 'days
    cases_by_patch ~ neg_binomial(mean = rho * projected, r = k)
  }
}
```

Generates one observation stream per patch.

### 12.4 Sampling vs Scoring

The `observations {}` block is evaluated at runtime in both directions.

- **Forward simulation** (`camdl simulate`): the runtime evaluates each stream's
  projection on its `emit_schedule` and **samples** from the declared likelihood
  family to produce synthetic observations. Synthetic-observation files are
  written when `--obs`, `--obs-dir`, or `--obs-only` is passed (see §21); no
  observation file is emitted by default. Trajectories are written independently
  via `--output` / stdout.
- **Inference** (`camdl fit` and friends): the runtime **scores** observed data
  against the same likelihood family, producing log p(y | θ). PGAS, IF2,
  particle filtering, and PMMH all consume the `observations {}` declarations
  via the compiled `dmeasure` / `rmeasure` paths. When fitting with `--data`,
  the data file's time column supplies the observation times and the declared
  `emit_schedule` is not consulted; the schedule is used only for forward
  synthetic-data generation under `simulate`.

The emission cadence is written `emit_schedule = every N 'unit` or
`emit_schedule = at [t1 'unit, t2 'unit, ...]` — the unit rides on each list
element (e.g. `at [7 'days, 14 'days]`), not after the bracket (§12 examples).
A bare `every`/`at`
field at the top of an observation block is the removed pre-gh#171 form and is
rejected with **E272** pointing at the `emit_schedule = ...` rewrite. Monthly
incidence can be obtained natively by setting `emit_schedule = every 30 'days`
(or `every 1 'months` once time-unit arithmetic is implemented).

---

## 13. Interventions

Deterministic state modifications at scheduled times. **Inactive by default.**
Enabled via scenarios or CLI.

```camdl
interventions {
  sia_round_1 : transfer(fraction = 0.80, from = S, to = V) at [180, 545]

  routine_vacc : transfer(fraction = vacc_rate, from = S, to = V) {
    every = 30 'days
    from  = 0 'days
    to    = 2 'years
  }

  importation_pulse : { I_child_p1 = I_child_p1 + 10  at = [90] }
}
```

### 13.1 Actions

There are three actions. `transfer` and `add` are written as function-call
forms; `set` (assign a compartment a value) is written inside the block form as
`COMP = EXPR` — there is no `set(...)` function:

```camdl
# transfer and add — function-call action forms
NAME : transfer(fraction = EXPR, from = COMP, to = COMP) at [...]   # move fraction
NAME : transfer(count = EXPR, from = COMP, to = COMP) at [...]       # move count
NAME : add(COMP, EXPR) at [...]                                       # add a count

# set — block form, one or more `COMP = EXPR` assignments plus a schedule
NAME : { COMP = EXPR  at = [...] }                                    # override value
```

`transfer` is atomic: `delta = floor(source * fraction)` computed from
pre-intervention state, then `source -= delta, dest += delta` applied together.

**Stratified compartments in actions.** `transfer(from = S, to = V)` with bare
compartment names expands over all strata (see §25.10). `set` and `add` instead
target a **single compartment by name**. On a stratified compartment the bare
name (e.g. `I`) is not a single compartment after expansion, so write the
**expanded stratum name** directly (`I_child_p1 = ...`, `add(I_child_p1, 5)`).
The compiler verifies both verbs' targets against the expanded compartment
table and rejects a stratified family or an unknown name with **E265**; the
family case lists the available cells. (Index-binder forms like `I[child, p1]`
on the left of a `set` are not part of the grammar.)

**Pairing rule for a bare stratified `transfer`.** The two endpoints pair
cell-for-cell, and may do so only when they are **declared with the same
dimensions, in the same order**. The compiler compares declared dimension
vectors, not expanded cell names, so two different dimensions that share level
names (`age = [low, high]` against `risk = [low, high]`) never pair silently. A
mismatch — one endpoint stratified and the other not, different dimensions, or
the same dimensions in a different order — is **E237**, which names both shapes.
A fully-indexed endpoint denotes one cell and carries no dimensions, so
`transfer(from = S[child], to = S[adult])` and `transfer(from = S[child], to =
V)` with `V` unstratified are ordinary single-cell transfers, not fan-outs. The
`fraction` is one expression shared by every cell; for coverage that varies by
stratum, write the indexed family form (§13.3).

**A bare endpoint inside an indexed family is an error.** `vacc[a in age] :
transfer(from = S, to = V)` would fan out over every cell *within each
instance*, transferring each cell once per instance — with `P` strata the
realised coverage is `1 − (1 − f)^P`, not `f`. That is **E239**; write
`from = S[a], to = V[a]`, or drop the `[a in age]` binder to fan out once.

**`count` does not fan out.** A fraction is scale-free, so the same value
applies to every cell. A count is absolute: applying it per cell would move
`count` individuals out of *each* stratum, multiplying the intended total by the
number of cells — 1548× on the national example in §25.10. `count` on a bare
stratified transfer is therefore **E238**. Write `fraction =`, or index the
transfer so each instance names one cell. `count` is unrestricted wherever the
transfer resolves to a single cell.

### 13.2 Scheduling

**Inline `at` form** (specific times, most common):

```
NAME : ACTION at [TIME, ...]     # times in model time_unit
```

**Block form** (recurring or complex schedules):

```
NAME : ACTION {
  every = DURATION             recurring interval
  from  = DURATION             start of recurring (default: t_start)
  to    = DURATION             end of recurring (default: t_end)
}
```

**In anchored mode**, `every`, `from`, and `to` must be classified `Exact`
(§2.1) — a `Calendar`-classified duration (e.g. `every = 1
'months`) is
**E322**, with a hint pointing at `every = 30 'days` for an affine ~monthly
recurrence or at an explicit calendar-listed `at [date(...), date(...), ...]`
schedule for true month-aligned recurrence. In unanchored mode the classifier is
inactive and `every = 1 'months` is fine.

### 13.3 Indexed Interventions

An intervention can be declared with an **index binder**, creating a **family**
of interventions — one per stratum — in a single line:

<!-- camdl-doctest-preamble: iv-patch
compartments { S, V, I, R }
dimensions {
  patch = [north, south, east]
}
stratify(by = patch)
parameters {
  beta     : rate
  gamma    : rate
  vacc_eff : probability
  sia_cov  : probability
}
let N[p in patch] = S[p] + V[p] + I[p] + R[p]
transitions {
  infection[p in patch] : S[p] --> I[p] @ beta * S[p] * I[p] / N[p]
  recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]
}
-->

```camdl preamble=iv-patch
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
(see §17).

Individual members can still be addressed by their expanded name (`"sia_north"`)
when fine-grained control is needed.

**Per-patch timing from a table.** When SIA rounds happen on different dates per
patch, store the schedule in a table and reference it in the `at` list:

```camdl
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

### 13.4 Activation

Interventions are off by default. Enable via scenarios or CLI:

```bash
camdl simulate model.camdl --enable sia_round_1 --seed 42
```

---

### 13.5 Events

Events are always-active scheduled state modifications. They share the same
action grammar and scheduling as interventions but fire unconditionally — they
cannot be disabled via scenarios.

Use events for structural demographic processes (cohort entry, seasonal
migration, importation seeding). Use interventions for policy choices (SIA
campaigns, school closures).

<!-- camdl-doctest-preamble: events-demo
compartments { S, I, R }
parameters {
  beta   : rate
  gamma  : rate
  cohort : probability
}
let N = S + I + R
forcing {
  birthrate : sinusoidal 'per_year {
    baseline  = 0.03
    amplitude = 0.0
    period    = 365.25 'days
    phase     = 0 'days
  }
  pop : sinusoidal 'count {
    baseline  = 100000.0
    amplitude = 0.0
    period    = 365.25 'days
    phase     = 0 'days
  }
}
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
-->

```camdl preamble=events-demo
events {
  cohort_entry : add(S, cohort * birthrate(t) * pop(t))
    every 365.25 'days at_day 251

  importation : add(I, 10) at [30]
}
```

Events support the same features as interventions: indexed events, `where`
guards, recurring schedules, and all action types.

### 13.6 The `add` Action

```
add(COMPARTMENT, EXPR)
```

Adds `round(EXPR)` individuals to COMPARTMENT. Accepts negative values
(outflow). If the result makes the compartment negative, a warning is emitted
but the simulation continues — in a particle filter, the particle gets a bad
trajectory and is resampled away.

### 13.7 The `at_day` Schedule

For `add` events and interventions that recur on a specific day within each
period. The `every … at_day …` schedule is available only on the `add` action
(`transfer`/`set` use the `at [...]` or `{ every = …; from = …; to = … }`
schedule forms instead):

```camdl
NAME : add(COMP, EXPR) every PERIOD at_day DAY
```

`at_day` is the absolute phase within the period, measured from `t = 0`. Fire
times are `at_day + k * period` for the smallest `k` where `target >= t_start`.
The engine fires on the single timestep where `|t - target| < 0.5 * dt`, so each
period fires exactly once as long as `dt` is no coarser than the period
(`dt <= period`). A coarser `dt` would round two consecutive targets onto the
same integrator step, silently dropping a fire; rather than merge them, the
engine rejects such a schedule at simulation start with a hard error (use a
finer `dt`, or widen the period).

Example: `every 365.25 'days at_day 251` fires on day 251 of each year. If
simulation starts at `t = 100`, the first fire is at `t = 251` (not `t = 351`).

This replaces manual `mod(t, period)` arithmetic, which silently double-fires
when `dt` does not evenly divide the period.

---

### 13.8 Balance Constraint

Forces one compartment to satisfy a population conservation constraint at every
substep. After all transitions, clamps, events, and interventions apply, the
target compartment is overwritten:

<!-- camdl-doctest-preamble: balance-demo
compartments { S, E, I, R }
parameters {
  beta  : rate
  sigma : rate
  gamma : rate
}
let N = S + E + I + R
forcing {
  pop : sinusoidal 'count {
    baseline  = 100000.0
    amplitude = 0.0
    period    = 365.25 'days
    phase     = 0 'days
  }
}
transitions {
  infection : S --> E @ beta * S * I / N
  progress  : E --> I @ sigma * E
  recovery  : I --> R @ gamma * I
}
-->

```camdl preamble=balance-demo
balance {
  R = pop(t) - S - E - I
}
```

This matches pomp's `R = nearbyint(pop) - S - E - I` pattern for models where
the population trajectory is externally specified and the birth/death rates
don't exactly reproduce it. The balance target is excluded from the
non-negativity clamp — a negative value signals a broken model.

Events that inject people without a source (e.g., `add(S, 20000)`) will increase
the compartment total. The balance compartment absorbs this by decreasing to
maintain the constraint.

### 13.9 Reactive Interventions

A reactive intervention fires as a function of what *surveillance has detected*,
not on a fixed calendar. It is a third fire source alongside scheduled
interventions (`at [...]`) and events: a policy whose timing the model
discovers at run time. The motivating case is outbreak response — "run an SIA
after AFP/ES detection crosses a threshold" — where a fixed schedule cannot
express the dependence on observed data.

<!-- camdl-doctest-preamble: reactive-demo
compartments { S, E, I, R, V }
parameters {
  beta                  : rate
  sigma                 : rate
  gamma                 : rate
  rho                   : probability
  afp_trigger_threshold : count
  sia_coverage          : probability
}
let N = S + E + I + R + V
transitions {
  infection   : S --> E @ beta * S * I / N
  progression : E --> I @ sigma * E
  recovery    : I --> R @ gamma * I
}
observations {
  weekly_afp {
    columns       { time : time, weekly_afp : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_afp    ~ poisson(rate = rho * projected)
  }
}
-->

```camdl preamble=reactive-demo
reactive_interventions {
  # Fire an SIA 21 days after the trailing-28-day AFP count crosses the
  # threshold, then rate-limit re-firing to once every 180 days.
  mop_up
    : when sum_observed(weekly_afp, window = 28 'days) >= afp_trigger_threshold {
        after    = 21 'days
        action   = transfer(fraction = sia_coverage, from = S, to = V)
        once     = false
        cooldown = 180 'days
      }
}
```

The `when` predicate is a boolean over **trigger inputs** — data visible to
policy, never latent model state:

- `observed(stream)` — the most recent observed value of an observation stream.
- `sum_observed(stream, window = D)` — the sum of `stream` over the trailing
  window `D`.

These are valid **only** inside a `when` predicate; using `observed(...)` in a
transition rate or any other model expression is an error (it would read data
the rate has no business reading). The predicate combines comparisons with
`and` / `or` / `not`; each comparison has exactly one trigger input on one side
and a constant or parameter threshold on the other.

The policy body fields:

| Field      | Meaning                                                            |
| ---------- | ----------------------------------------------------------------- |
| `action`   | The state change to apply — same `transfer` / `add` grammar as interventions. |
| `after`    | Non-negative lag between the trigger firing and the effect (default `0`). |
| `once`     | `true` (default) fires at most once; `false` allows repeats.      |
| `cooldown` | Minimum time between firings when `once = false`. Mutually exclusive with `once = true`. |

**Engine semantics.** The trigger predicate is a **level** test evaluated at
each observation-emission boundary (not continuously), gated by `once` and
`cooldown`. A windowed reducer `sum_observed(stream, window = D)` folds the
emissions in the half-open trailing window `(now − D, now]` — open on the left,
closed on the right, so an emission landing exactly at `now` is included. When a
policy fires, its effect is enqueued at `trigger_time + after`, and `cooldown`
is measured from the **trigger time** (when the predicate crossed), not from the
delayed effect time.

The trigger always reads **reported surveillance** — the realized observation
draw at each emission boundary — never latent model state. Reactive policies run
only on the chain-binomial *forward* backend, which has no particle ensemble
(and never runs in inference; §13.9 Status), so there is nothing "shared across
particles" today. A future particle-local `scope` reading latent state is
deferred; until then there is no `scope` key.

A reactive intervention is a **policy** (like `interventions {}`, not `events {}`):
it is scenario-toggleable, so a `baseline` scenario can omit it and a
`with_response` scenario can `enable` it, which is exactly how prospective
policy analysis compares responding vs non-responding worlds.

**Reactive interventions are policy interventions and are inactive by default.**
Enable them with scenarios or `--enable`, exactly like `interventions {}`. A
reactive policy that is not enabled is dropped from the run (the same toggle
semantics as a scheduled intervention), so a plain `simulate` with no scenario
runs the baseline without it.

> **Status (gh#204).** Reactive interventions are parsed, dimension-checked, and
> represented in the IR. Forward simulation on the **chain-binomial** backend
> executes the agenda: an enabled policy fires when its trigger crosses, after
> the `after` lag, honouring `once`/`cooldown`; each firing is recorded in the
> run's `reactive_log.tsv` artifact. Inference (IF2/PGAS/PMMH) and the
> Gillespie/ODE forward backends do **not** yet run reactive policies — an active
> reactive policy there fails with a clear `REACTIVE_INTERVENTIONS` capability
> error. A dormant (unenabled) reactive policy is inert, so a run that does not
> enable it is accepted on every backend. The DSL and IR surface are stable.

### 13.10 Within-substep effect ordering

When several effects land at the same time step they apply in a fixed,
backend-shared order, so a modeller can predict the state an observation or a
later effect sees. For the fixed-step backends (chain-binomial and the
discrete-time filters) the order is:

1. **Inflow events** (`add`, source-less `--> D` events) are computed from the
   **start-of-step snapshot** and fused into the transition draw.
2. **Transition draws** advance the compartments (Euler-multinomial,
   independent-Poisson, RK4, or SSA, per backend).
3. **Residual events** — draining `transfer` events and `set` — apply to the
   **post-transition** state.
4. **Scheduled interventions** apply next, on the post-advance state, in
   declaration order.
5. **Balance** (chain-binomial only) overwrites its target last.
6. The **non-negativity check** runs (the balance target is exempt — a negative
   there signals a broken model, reported separately, not by this check).
7. The trajectory row for this step is recorded **after** all of the above.

The consequence a modeller must know: recorded output is **post-effect**. An
observation emitted at a time when an intervention fires — a vaccination
`transfer`, say — sees the **post-vaccination** state, not the pre-vaccination
state.

---

## 14. Timepoints and Reserved Identifiers

> **Partially implemented.** The `timepoints { }` block is parsed but the
> declared timepoint values are currently discarded by the expander and not
> available in expressions. Full timepoint support is not yet implemented. The
> built-in reserved identifiers `t_start` and `t_end` are always available
> regardless.

```camdl
timepoints {
  midpoint     = 1 'year
  intervention = 180 'days
}
```

### 14.1 Built-in Timepoints

`t_start` and `t_end` are **reserved identifiers** automatically defined from
the `simulate` block, holding the simulation's start and end times. They are
available in any expression — most usefully to anchor intervention or event
schedule windows relative to the run (e.g. `from` / `to`).

If `simulate` is absent (e.g., during `camdl check`), the expander silently
defaults `t_start = 0` and `t_end = 100`; expressions referencing them use those
defaults. There is no warning for a missing `simulate` block.

### 14.2 Reserved Identifiers

Three distinct mechanisms prevent a name from being used as a declaration. They
fail differently, so they are worth separating.

**1. Genuinely reserved names** — checked explicitly by the compiler. Declaring a
compartment, parameter, or `let` binding with one of these is an **E100** error
("name '…' is reserved …"):

```
t          # current simulation time
t_start    # simulation start time (from simulate block)
t_end      # simulation end time (from simulate block)
dt         # current substep length (used inside rate expressions)
pi         # the constant π
e          # Euler's number
```

**2. Keywords** — the lexer tokenizes these, so they cannot appear as an
identifier; using one as a name is a bare **E001** syntax error, not a
reserved-id diagnostic. This set includes the block keywords (`compartments`,
`parameters`, `tables`, `forcing`, `transitions`, `observations`,
`interventions`, `events`, `reactive_interventions`, `ode`, `output`,
`simulate`, `init`, `scenarios`, …), the type keywords (`rate`, `probability`,
`positive`, `count`, `real`, `integer`, `instant`, `duration`), and
operator/iteration keywords (`sum`, `consecutive`, `where`, `let`,
`if`/`then`/`else`, `and`/`or`/`not`, `in`, `by`, `from`, `to`, `every`,
`until`, `at`, `when`, `action`, `origin`, `columns`, `emit_schedule`, …).

**3. Function and distribution names** — these are **not** reserved and **not**
keywords. They are recognized only in call position; used as a parameter name
they compile fine. This includes the calendar builtins (`add_calendar_months`,
`add_calendar_years`, `date`, `date_range` — note only the `_months`/`_years`
calendar adders exist; there is no `add_calendar_days`/`add_calendar_weeks`), the
rate wrappers (`overdispersed`, `deterministic`), the likelihood distributions
(`poisson`, `neg_binomial`, `normal`, `binomial`, `beta_binomial`, `beta`,
`bernoulli`, `diagnostic_test`), the observation projection name (`projected`), and the
scenario names (`baseline`, `scenario`). Reusing one as an ordinary parameter is
legal but inadvisable for readability.

A genuinely-reserved name (group 1) produces:

```
ERROR E100: parameter name 't_end' is reserved for simulation time
```

---

## 15. Initial Conditions

### 15.1 Un-Stratified Models

<!-- camdl-doctest-preamble: init-sir
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
  N0    : count
  I0    : count
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
-->

```camdl preamble=init-sir
init {
  S = N0 - I0
  I = I0
}
```

Unlisted compartments default to 0. Expressions can reference parameters.

### 15.2 Stratified Models

When compartments have index dimensions, **bare names are a compile error.** The
compiler cannot guess how to distribute a total across strata.

```camdl
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

**Indexed parameter references** work in init RHS expressions. If `N0[patch]` is
an indexed parameter, both the mangled form and the indexed form are accepted:

<!-- camdl-doctest-preamble: init-region
compartments { S, I, R }
dimensions {
  patch = [urban, rural]
}
stratify(by = patch)
parameters {
  beta  : rate
  gamma : rate
  N0[patch] : count
  I0    : count
}
let N[p in patch] = S[p] + I[p] + R[p]
transitions {
  infection[p in patch] : S[p] --> I[p] @ beta * S[p] * I[p] / N[p]
  recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]
}
-->

```camdl preamble=init-region
init {
  S[urban] = N0[urban] - I0   # indexed syntax — preferred
  S[rural] = N0_rural         # mangled form — still accepted
}
```

Named indexing works in init:

<!-- camdl-doctest-preamble: init-strat
compartments { S, I, R }
dimensions {
  age   = [child, adult]
  patch = [p1, p2]
}
stratify(by = age)
stratify(by = patch)
parameters {
  beta  : rate
  gamma : rate
}
let N[a in age, p in patch] = S[a,p] + I[a,p] + R[a,p]
transitions {
  infection[a in age, p in patch] : S[a,p] --> I[a,p] @ beta * S[a,p] * I[a,p] / N[a,p]
  recovery[a in age, p in patch]  : I[a,p] --> R[a,p] @ gamma * I[a,p]
}
-->

```camdl preamble=init-strat
init {
  S[age = child, patch = p1] = 100000
}
```

Unlisted stratum combinations default to 0. For a 774-patch model, only the
patches mentioned in init are nonzero — the rest start empty. This is common for
initialization from a single-patch seeding event.

### 15.3 Init from Tables

For large spatial models where per-stratum populations come from a CSV, declare
a table (§6) and reference it directly in init expressions:

<!-- camdl-doctest-preamble: init-table
compartments { S, I, R }
dimensions {
  patch = [north, south]
}
stratify(by = patch)
parameters {
  beta  : rate
  gamma : rate
}
let N[p in patch] = S[p] + I[p] + R[p]
transitions {
  infection[p in patch] : S[p] --> I[p] @ beta * S[p] * I[p] / N[p]
  recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]
}
-->

<!-- camdl-doctest-data: data/population.tsv
patch	N0
north	100000
south	80000
-->

```camdl preamble=init-table
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

This is fully supported. Per-stratum initial values come from index binders and
compile-time table lookups; there is no `distribute(...)` allocation helper.

### 15.4 Drawn Initial Conditions

An initial condition may be **drawn** rather than computed. Where `=` says "this
compartment starts at this value", `~` says "this compartment starts at a draw
from this distribution" — the same reading `~` has for a parameter prior (§4)
and for an observation likelihood (§9).

<!-- camdl-doctest-preamble: init-drawn
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
  N0    : count
  I0    : count
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
-->

```camdl preamble=init-drawn
init {
  I ~ poisson(rate = I0)   # the number introduced is a DRAW around I0
  S = N0 - I               # reads the drawn I, so S + I = N0 exactly
}
```

The entries are evaluated in **dependency order**, so `S` reads the value `I`
was drawn as, not the value it was expected to be. The population budget
therefore holds on every draw with no `balance {}` block.

Which laws are admissible is decided by the compartment's kind:

| compartment | admissible laws                                                                  |
| ----------- | -------------------------------------------------------------------------------- |
| integer     | `poisson(rate = ..)`, `binomial(n = .., p = ..)`, `neg_binomial(mean = .., r = ..)` |
| `: real`    | `normal(mean = .., sd = ..)`                                                      |

A mismatch is a compile error (E344), as is a law the initial-state vocabulary
does not have (E343) — `bernoulli`, `beta` and `beta_binomial` describe a
*measurement* of a compartment, not the compartment itself, so they belong in
`observations {}` and not here. `binomial`'s `p` must be a `probability`-kinded
parameter; a `count` there is E344 with the parameter named.

`neg_binomial` is the choice when introductions are **clustered** rather than
independent — the dispersion `r` controls how much more variable the count is
than a Poisson of the same mean, and large `r` recovers Poisson.

A law may not be declared on the `balance {}` target (E345): the balance stage
recomputes its compartment after every substep, so the draw would be overwritten
before the first step is taken.

**What a drawn initial condition does at fit time.** Under `pgas` it becomes a
term of the target — `log p(x₀ | θ)` appears in the complete-data likelihood and
in the `initial_state_ll` column of each chain's trace — so the law's parameters
are estimated from the data rather than fixed. Under `if2` each particle draws
its own initial state. `pfilter` and `pmmh` refuse a model with a drawn initial
condition: their bootstrap filter evaluates one initial state and copies it to
every particle, which would condition the whole swarm on a single realization of
`x₀`. The deterministic (ODE) fits start every compartment at its law's mean.

---

## 16. Output and Quantities

A simulation writes a **trajectory** — the time series of compartment states,
sampled on a schedule. With no `output {}` block the default schedule applies;
declare one to set the cadence or give explicit output times.

> **Default schedule.** Snapshots every `1` in the model's `time_unit`, covering
> `[t_start, t_end]` — where the window is taken from the `simulate {}`
> block (or `(0, 100)` if `simulate {}` is omitted). The simulate command writes
> the trajectory to `--output` (or stdout) and writes observation files only when
> `--obs` / `--obs-dir` / `--obs-only` is passed.

```camdl
output {
  trajectories {
    every  = 0.5 'days     # regular cadence (sub-unit is fine for fast dynamics)
  }
}
```

The schedule mirrors the observation surface: use **either** `every = E` for a
regular cadence **or** `at = [t1, t2, ...]` for an explicit list of output times
— the two are mutually exclusive (specifying both is an error). A `format = …`
field parses but is currently inert: the writer always emits wide TSV. (`format`
is stripped before run hashing, so it never affects cached results.)

```camdl
output { trajectories { at = [0, 30, 60, 90] } }   # snapshot only at these times
```

### 16.1 Output Files

```
trajectories.tsv      # time × compartment states (one row per output time)
metadata.json         # run provenance (see §19)
```

### 16.2 IR Mapping

The trajectories block compiles to the IR `output` schedule: `every = E` →
`OutRegular { start, step }` (start defaults to `t_start` so the schedule covers
exactly the requested window `[from, to]`, including anchored models with a
negative `t_start`); `at = [...]` → `OutAtTimes`. The runtime writes the
trajectory directly during simulation.

Output emission is confined to `[start, simulation.t_end]`: `simulation.t_end`
is the sole horizon authority, and output times are derived from it at emission
— a regular schedule enumerates up to `t_end`, and an explicit `at = [...]` time
beyond the horizon is not emitted (never against a frozen post-horizon state).

Synthetic observations (forward simulation) are not part of the trajectory
output; they are produced by the simulate command's `--obs` family of flags
(§21), generated by the runtime's `sample_observations` method from the
observation model definitions.

### 16.3 Per-run overrides (CLI and config)

The cadence and the columns can also be set per run, without editing the model
— useful for exploratory runs and for large stratified/spatial models whose
`flow_*` columns dominate the output. The `simulate` flags and `batch.toml`'s
`[output]` section are the same option set (one shared definition), so the
surface is identical. (Only `simulate` and `batch.toml` write trajectories, so
the view applies there; `fit.toml` has no trajectory output.)

| Flag                | `[output]` key       | Effect                                                                                                                            |
| ------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `--output-every N`  | `every = N`          | One row every `N` time-units, overriding the model's `output { every }`. A plain number in the model's `time_unit` (like `--dt`). |
| `--no-flows`        | `no_flows = true`    | Drop every `flow_*` column.                                                                                                       |
| `--columns A,B,…`   | `columns = ["A","B"]`| Restrict to this allow-list of output column names (compartments and/or `flow_<name>`); emitted order follows the model.          |

```bash
# weekly latent state, no flow columns — on a 23-patch spatial run the
# patch×patch flow columns are ~93% of the output
camdl simulate polio.camdl --draws prior -n 100 --output-every 7 --no-flows
```

```toml
# batch.toml
[output]
every    = 7
no_flows = true
# columns = ["S", "I_c", "I_v", "R"]
```

These overrides participate in run identity. `--output-every` rewrites the
model schedule (it rides the model digest, re-keying only runs that use it);
`--no-flows` / `--columns` ride the `config` level, because a column subset is a
distinct, reproducible artifact — a content-addressed leaf cannot share a
`run_id` with the full one. (Introducing this view bumps the `config` schema
version, so all sim `run_id`s shift once — a deliberate, versioned turnover;
existing cached sims re-run on next use.) An unknown `--columns` name is a hard
error that lists the valid columns. (The
`[design.*]` batch path honors `every` but not `no_flows` / `columns` yet, and
rejects the latter loudly.)

### 16.4 Derived quantities (`quantities {}`)

`quantities {}` is a top-level block — a sibling of `observations {}` and
`output {}`, not nested in either — that declares **derived quantities**:
summaries computed from a run and reported alongside the trajectory, but never
scored as data. It is the non-scored twin of an observation. Where
`observations {}` defines what the likelihood _sees_, `quantities {}` defines
what you want _read back_ — a peak size, an attack rate, the time an outbreak
takes off — so a summary no longer has to be smuggled through a fake scored
stream.

<!-- camdl-doctest-preamble: quantities-demo
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
  rho   : probability
  k     : positive
  N0    : count
  i_thr : count
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases ~ neg_binomial(mean = rho * projected, r = k)
  }
}
-->

```camdl preamble=quantities-demo
quantities {
  prevalence      = I / N                    # series — one value per output time
  attack_rate     = final((N0 - S) / N0)     # scalar reduction
  peak_prevalence = max(I / N)               # scalar — unary max is a peak reduction
  time_to_peak    = time_of_max(I)           # a time (a date in an anchored model)
  takeoff         = first_above(I, i_thr)    # first time I exceeds the threshold i_thr
  fadeout         = last_above(I, 0)         # last time I is above 0
  outbreak_dur    = fadeout - takeoff        # arithmetic over already-reduced scalars
  peak_reported   = max(observations.cases)  # reduce a simulated observation stream
  size_at_day_30  = value_at(N0 - S, 30)     # the series value at a stated time
  outbreak_size   = value_at(N0 - S, last_obs)  # … at the end of observed data
}
```

Each binding is `name = <expr>`. Whether a quantity is a **series** or a
**scalar** is determined by the expression, never declared:

- **Series** — an expression over compartment state (with parameters, `time`,
  tables, forcings) and **no reduction** yields one value per output time
  (`prevalence` above), on the trajectory's output schedule (§16).
- **Scalar** — a **reduction** collapses a series to a single number or a single
  time. The temporal reductions are `final`, `mean`, `integral`,
  `count_above` / `count_below`, `value_at`, `time_of_max` / `time_of_min`, and
  `first_above` / `first_below` / `last_above` / `last_below`. In addition, a
  **unary** `max` / `min` reduces a series to its peak / trough — binary
  `max(a, b)` / `min(a, b)` stay pointwise operators everywhere else.

**Reducing a simulated observation.** A reduction may fold a simulated
observation instead of raw state: `max(observations.<stream>)` reduces the same
`y_sim` the run drew for `<stream>`, never triggering a fresh draw — so the
quantity and the emitted observation file always agree. An observation source
**must** be reduced; a bare `observations.<stream>` series is rejected
(**E289**), because a quantity that re-sampled observations on every read would
not be reproducible.

**Reduction arithmetic.** Already-reduced scalars combine with `+ - * /` and
comparisons, as in `outbreak_dur = fadeout - takeoff`. A reduction call may not
be **nested** inside that arithmetic — `last_above(I, 0) - takeoff` is rejected
(**E289**); bind the reduction to its own quantity first, then combine the
names. A reduction name used in a _rate_ or a `let` binding (outside
`quantities {}`) is rejected (**E290**): a reduction summarizes a whole run and
is meaningless inside a per-step propensity.

**Reading a series at a time.** `value_at(series, TIME)` is the series value
at the last output time at or before `TIME` (the state *as of* `TIME`; values
are never interpolated). `TIME` is a constant time expression —
`date("2026-08-10")` under a declared `origin`, or a number in model time —
**or an observation anchor**: `last_obs` (the end of observed data),
`first_obs` (its start), each optionally plus or minus a constant duration.

```camdl
outbreak_size  = value_at(N0 - S, last_obs)
a_week_earlier = value_at(N0 - S, last_obs - 1 'weeks)
at_the_start   = value_at(N0 - S, first_obs)
```

An anchor's *value* is data-dependent and resolved where data is in hand
(`fit predict`); the offset is **not** — it is folded to model time units at
compile time, so the compiled model itself stays data-independent. A forward
`simulate` of an anchored model is rejected with an error naming the quantity,
because a simulation has no observed data to anchor to.

The offset must carry a duration unit: a bare `last_obs - 7` is rejected
(**E335**), because there is no unit to interpret. Under a declared `origin` a
`'months`/`'years` offset is rejected too (**E321**) — an anchor is an instant,
and a calendar month is not a fixed number of days, so the resolved time would
move with the data. Note that camdl writes durations with a leading tick
(`1 'weeks`); a bare word (`1 weeks`) is **E115**, which names the tick form.
The command line takes the opposite convention (`--to "last_obs + 8 weeks"`,
bare words), because a tick is a shell-quoting hazard.

An anchor is legal **only** as the whole time argument, optionally `±` a
constant duration; folding it into a larger expression (`2 * last_obs`) is
rejected (**E335**), and outside its granted positions it is an ordinary
unknown name.

**Censoring.** A timing reduction that never resolves — `first_above` on a draw
that never crosses the threshold, `time_of_max` on an all-zero series — is
reported as **right-censored**, not as a fabricated time. A `value_at` whose
`TIME` falls outside the draw's trajectory window is censored the same way,
never clamped to the window edge: clamping would silently report the value at
the horizon instead of the value asked for.

**Where they run, and run identity.** Quantities run wherever a simulation does:
over prior-predictive draws (`simulate --draws`), over a fitted posterior
(`fit predict`), and in a plain `simulate`. Banded results land in
`quantities/<name>.tsv` with a `quantities.json` manifest describing each
quantity's kind. Because they are _derived reports_ computed from a run rather
than inputs to it, adding or changing a `quantities {}` block never re-keys a
model's `run_id`.

**`#'` documentation carries the same guarantee.** A doc comment is presentation
metadata: the compiler emits it into the IR envelope's `docs` dictionary, which
sits *outside* the `model` object that run identity is computed from. Rewording
a docstring — correcting a citation, sharpening a caveat — therefore leaves
`model_identity` unchanged and orphans no completed fit. Both rules exist so
that the two edits a modeller most often wants to make *after* a long run, the
reporting vocabulary and the scholarly record of why a prior is what it is, are
free to make. What *does* re-key is anything the model computes with: a rate, a
prior, a bound, a compartment, a transition, an observation.

**A vocabulary can live in its own file.** A `quantities {}` block is a
_reporting vocabulary_, and several models often want the same one. Rather than
copy it into each model — where the copies drift — put it in an ordinary
`.camdl` file containing **only** a `quantities {}` block, and supply it at the
point of use:

```
camdl simulate model.camdl --quantities reporting/national.camdl --quantities-out out/
camdl fit predict @jigawa-baseline --quantities reporting/national.camdl
```

The file is compiled as a second compilation unit of that model's compile, so
its body is resolved against the model's own compartments, parameters and `let`
bindings, and a name the model does not declare is an error naming both the
name and the file. It **replaces** the model's own block; it never merges,
because a merge rule would make the reported table depend on which of two files
declared a name first. A file that declares anything besides quantities is
**E339**; one that declares no quantities is **E340** (replacement means an
empty vocabulary would report nothing).

The emitted tables are keyed by the file's contents:
`quantities-<key>/<name>.tsv` with a matching `quantities-<key>.json` whose
`vocabulary` object records the file's path and digest. Two vocabularies applied
to one run therefore produce two tables rather than overwriting one, and
correcting a formula in place produces a new table rather than a stale cache
hit. Run identity is untouched, as above.

`fit predict --quantities` is the only way to change what an EXISTING fit
reports: `fit predict` reads the model IR archived inside the fit, so editing a
formula in the model source has no effect on a fit that has already run. The
vocabulary is compiled against the fit's model source and refused unless that
source is still the same model the fit ran on — quantities are excluded from
that comparison, so a reporting-only edit to the source is not a mismatch.

---

## 17. Scenarios

Patch-based modifications to the baseline. Baseline is the identity patch — the
model as defined, no modifications.

<!-- camdl-doctest-preamble: scenario-sia
compartments { S, V, I, R }
parameters {
  beta    : rate
  gamma   : rate
  sia_cov : probability
}
let N = S + V + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
interventions {
  sia : transfer(fraction = sia_cov, from = S, to = V) at [180]
}
-->

```camdl preamble=scenario-sia
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

### 17.1 Patch Operations

```
label   = STRING                   human-readable name for the scenario
enable  = [INTERVENTION, ...]      turn on interventions
disable = [INTERVENTION, ...]      turn off interventions
set     = { PARAM = EXPR, ... }    override parameter values
scale   = { PARAM = FACTOR, ... }  multiply (compiler checks domain validity)
compose = [SCENARIO, ...]          apply patches in sequence
simulate { to = EXPR }             override only the end time (§17.2); `to` is required, and from/dt/integrator are model-wide, not per-scenario (E106)
```

A scenario may override the end time and nothing else. The rule is that a
scenario overlays what leaves the **trajectory prefix** intact: extending or
truncating `to` never re-tiles `[from, old_to]`, so two scenarios differing only
in horizon stay byte-identical over their shared span and the paired-seed
coupling (§3.1 of the run spec) survives. `dt` and `integrator` re-tile the
substep grid, so arms would diverge from `from` for purely numerical reasons and
any between-arm difference would mix in discretization error; `from` is the same
class, with the extra wrinkle that `init {}` is evaluated at `t_start`. All
three are `E106`, as is a `simulate {}` block that omits `to` — including an
empty one, which cannot mean anything.

Under an explicit `at = [...]` output list (§16), emission is exactly the listed
times within the horizon, so a scenario `to` is **inert** when it selects the
same entries the model horizon already would — no listed time falls between the
two. The compiler warns (`W106`) in that case only: a `to` that pulls in (or
drops) a listed time changes both the trajectory and every `quantities {}`
reduction, and is not warned about.

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

**Patch algebra.** Within one scenario, `set` applies before `scale`, so `scale`
multiplies the value `set` produced (or the inherited value, when the parameter
isn't `set`). Across a `compose` list, the composed sub-scenarios apply in listed
order and the scenario's **own** patch applies last, so it wins any collision
with a composed one. In `enable`/`disable`, an explicit `disable` beats an
`enable` of the same intervention — a name appearing in both lists ends up
disabled.

### 17.2 Scenario Inheritance — `extends`

A scenario can inherit from another via `extends = <parent_name>`, which is
**compile-time sugar**: the child is resolved as the parent with the child's
fields layered on top. The IR keeps its flat preset shape — downstream consumers
see no trace of inheritance.

<!-- camdl-doctest-preamble: scenario-vacc
compartments { S, V, I, R }
parameters {
  beta  : rate
  gamma : rate
  N0    : count
  I0    : count
}
let N = S + V + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
interventions {
  vaccination : transfer(fraction = 0.5, from = S, to = V) at [15]
}
-->

```camdl preamble=scenario-vacc
scenarios {
  baseline {
    label = "No intervention"
    set = {
      beta  = 0.4
      gamma = 0.15
      N0    = 10_000
      I0    = 10
    }
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

| Field                          | Behavior                                                       |
| ------------------------------ | -------------------------------------------------------------- |
| `set`, `scale`                 | Child keys override parent keys on collision; union otherwise. |
| `enable`, `disable`, `compose` | Parent + child, deduped preserving first-seen order.           |
| `label`, `simulate.to`         | Child overrides parent when present.                           |

**Expression scope.** Child `set` expressions are evaluated _after_ parent's
`set` is resolved — so `set = { beta = beta * 1.5 }` in a child reads the
parent's resolved `beta`. There's no default-at-declaration path in camdl; the
name must resolve to a concrete upstream value or the compiler errors.

::: {.callout-warning} **`enable`/`disable`/`compose` append parent's list to
the child's.** A child writing `enable = [masking]` under a parent with
`enable = [vaccination]` gets `[vaccination, masking]`, **not** just
`[masking]`. To remove a parent's intervention in a child, use `disable`. The
compiler emits **W310** whenever this merge actually changes the child's
declared list, so the surprise is observable rather than silent. :::

**Diagnostics:**

- **E25x** — cycle in `extends` chain (includes the full chain in the message).
- **E25y** — unknown parent scenario (suggests the closest name by edit
  distance).
- **E25z** — chain depth > 5; treat as a code smell and factor common ancestors,
  or request multi-parent composition as a future feature.
- **W310** — append-dedup of parent's enable/disable/compose changed the
  resolved list (see callout above).

### 17.3 Scenario Expression Scope

Inside `set = { PARAM = EXPR }`, the RHS expression can reference:

- The parameter's **current value** (its name refers to the pre-patch value)
- Other parameters (their pre-patch values)
- Literal constants

Compartment state, time, and other scenario settings are NOT in scope — scenario
patches are static transformations of parameter values, not runtime-dependent
operations.

---

## 18. Simulation Configuration

<!-- camdl-doctest-preamble: sir-basic
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
-->

```camdl preamble=sir-basic
simulate {
  from = 0 'days
  to   = 2 'years
  dt   = 0.5 'days     # optional: the discretization step
}
```

The `simulate {}` block sets the time window (`from`, `to`) and, optionally, the
discretization step `dt`. All three are unit-aware like any time literal:
`dt = 0.05 'months` is one month-scaled step (affine 30.44-day months), the same
convention `from`/`to` use.

`dt` is a **model knob**. A stochastic compartmental model's behaviour is
genuinely sensitive to the step — discretization error shrinks as `dt → 0`, and
Richardson-extrapolation diagnostics deliberately vary it — so the chosen step
is part of the model, declared next to the dynamics it discretizes. (`dt` is the
per-substep length the chain-binomial and ODE backends integrate
with; the exact-event Gillespie backend ignores it.)

The CLI `--dt` flag is the **override**: it wins over the model's `dt` for a
single run, which is exactly what a sensitivity sweep or a convergence check
wants. When neither the model nor `--dt` sets a step, it defaults to `1` in the
model's `time_unit`. Omit `dt` from the model only when you intend the run — not
the model — to choose it.

The optional `integrator` key selects the ODE method (it has no effect on the
chain-binomial or Gillespie backends). It is **tagged**: `integrator = rk4` is
fixed-step classic RK4 (the default — omit the key for it), and
`integrator = rk45 { atol = 1e-8  rtol = 1e-6 }` is adaptive Dormand–Prince
RK4(5), which takes large steps through smooth stretches and small steps only
where the trajectory moves fast. The tolerances `atol`/`rtol` are
**dimensionless** (error tolerances, not times), optional (omitted → the
runtime's calibrated default of `1e-8`/`1e-6`), and are **keys of the `rk45`
block** — they cannot be written without it, so an orphan tolerance (a tolerance
without `rk45`, or a tolerance on `rk4`) is a compile error. A model that
references `dt` in a rate (`Expr::Dt`) is incompatible with `rk45` — adaptive
stepping has no single fixed `dt` — and is rejected with a hard error pointing
back to `rk4`. The CLI `--integrator rk4|rk45` flag overrides the method for a
single forward `simulate` run (it preserves any model-declared tolerances);
there is no fit-side flag, because on the inference path the integrator is part
of the model's content identity and is declared in `simulate {}`.

A typo'd or unsupported key in `simulate {}` is a hard error (`E106`), never
silently dropped; the accepted keys are `from`, `to`, `dt`, `integrator`.

Seed is always external (CLI `--seed`), never in the model file.

### An observation-anchored horizon

`to` may be an **observation anchor** — `last_obs` or `first_obs`, optionally
`±` a constant duration — instead of a literal time. A forecasting model whose
horizon is "eight weeks past the end of the data" then states that, rather than
carrying a date that has to be re-typed on every data release:

```camdl
simulate { from = 0 'days  to = last_obs + 8 'weeks }
```

A scenario may anchor its own horizon the same way
(`scenarios { forecast { simulate { to = last_obs + 8 'weeks } } }`).

The horizon is resolved once per run, from the observation times the run binds,
and the resolved value is printed on stderr and substituted into the model the
run is content-addressed by — so a run under one data vintage is never served
from the store for another. The same rules as every other anchor position apply
(whole term, offset carries a duration unit, no `'months`/`'years` under an
`origin`).

Every command that binds observation data resolves anchors this way:
`camdl simulate --fit <fit.toml | fit run dir>`, `fit run`, `fit predict`, and
the three fixed-parameter commands `pfilter`, `profile` and `survey`, which fold
the window from whatever they bind through `--data` or `--fit`. A command with no
data cannot resolve one and refuses, naming the anchored construct.

`pfilter`, `profile` and `survey` score at the observation times rather than over
the model horizon, so a *scenario's* own `simulate { to }` still cannot be
honoured there and is refused rather than silently dropped. The comparison is
made after resolution, so a scenario anchor that resolves to the same time as the
model's is a no-op and runs.

Because the horizon is now unknown at compile time, three constructs that
**bake** it are refused rather than left to silently mis-fire:

- A recurring intervention or event with no `to` of its own (**E336**) — its
  window would default to the model horizon. Give the schedule its own `to`.
- The `every … at_day …` schedule form (**E337**), which has no `to` key at all
  to override with.
- A reactive policy (**E338**). Its monitoring window is fixed at compile time
  from the horizon, so under an anchored `to` the dynamics would run to the
  resolved end while the policy stopped reacting at the baked one — with no
  error. Until that is re-derived from the run horizon, an anchored model
  horizon and a reactive policy cannot be combined.

For the third case, and for any model that must keep a literal horizon, the
per-run alternative is the CLI: leave `simulate { to }` literal and pass
`camdl simulate --to "last_obs + 8 weeks"` (note the CLI's bare-word units).

---

## 19. Content-Addressable Output

Outputs are stored in a **content-addressed store** under an `output_dir` you
choose, partitioned by artifact kind (`sims/` for `simulate` and `batch`
trajectories, `fits/` for inference stages, and so on):

```
{output_dir}/
  sims/
    {model}/{config}/{params}/{scenario}/{seed}/
      traj.tsv
      run.json
```

A `simulate` trajectory's address is a **factored tuple of five levels** —
model, config, params, scenario, seed — nested in that order. Each path segment
is `{label}-{hash8}`, e.g. `sir_basic-3a7f2c1d/chain_binomial-dt1-1fb03eee/…`:
the `label` is a human-readable provenance tag (the model stem,
`chain_binomial-dt1`, `seed_1`) and the `hash8` is the first eight hex chars of
that level's structural content hash.

Navigation and display read `run.json`, never the segment text, so the label is
cosmetic. Renaming a scenario changes the directory name but not the identity,
which produces a harmless cache miss (a fresh directory, one redundant re-run),
never a wrong answer. There is **no** `00000000` special-case marker for an
empty (baseline) scenario: an empty `enable`/`disable`/`set` hashes to a
concrete level hash like any other input.

### 19.1 Run identity

A leaf's identity is its `run_id`: a single structural hash over the ordered
list of the five per-level content hashes,

```
run_id = hash(HASH_VERSION, kind, [model, config, params, scenario, seed])
```

recorded in full (64 hex chars) in `run.json`; the 8-hex path segments are a
readable factoring of it. Each level hashes a **disjoint slice** of the resolved
input set, and their union is the whole input:

| Level      | Covers                                                                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`    | the structural IR — compartments, transitions, observations, ODE equations, tables, `init`, `time_unit` — presentation-normalized (output format and time rendering are stripped, since they don't change results) |
| `config`   | backend, `dt`, `from` / `to`, the output schedule, the written-column selection (`--columns` / `--no-flows`), and degenerate-rate handling                                 |
| `params`   | the resolved base parameter values and any loaded table data                                                                                                              |
| `scenario` | the scenario **delta** only: the `enable` / `disable` lists and the `set` / `scale` parameter patch                                                                        |
| `seed`     | the process and base RNG seeds                                                                                                                                             |

Two runs that share a structural model share the `model` level even if one bakes
in calibrated values and the other supplies them externally; changing a
parameter value re-keys the `params` level while the `model` level is untouched.
The written-column selection is part of `config`, so two runs that request
different columns are distinct leaves — a leaf's bytes and its `run_id` never
disagree.

**Collision handling.** Eight hex chars per segment suffices because a collision
requires *every* level on the path to collide at once; the full 64-char hashes
in `run.json` are the authoritative check, and the store escalates a genuine
path-prefix collision by suffixing the final segment.

### 19.2 Cache reuse

A run is a **cache hit** when its factored path already exists. Because the
levels nest independently, an unchanged level's subtree is reused verbatim:

| What changed                        | Re-keys              | Reuse                                                       |
| ----------------------------------- | -------------------- | ---------------------------------------------------------- |
| model / IR                          | `model` + all below  | none                                                       |
| base params or table data           | `params` + below     | none                                                       |
| backend, `dt`, or column selection  | `config` + below     | none                                                       |
| one scenario's enable/disable/patch | that `scenario` leaf | sibling scenarios reused                                   |
| add more seeds                      | new `seed` leaves    | existing seeds reused                                      |
| rename a scenario (same delta)      | nothing (label only) | identity unchanged, but the renamed path is a fresh directory → one harmless re-run |

### 19.3 Enumerating runs

There is no manifest file. Every completed run is its own content-addressed
`run.json` leaf; to enumerate them, walk those leaves or read the derived
`index.json`. `camdl list` does this live.

### 19.4 Caching

Same inputs → same hashes → run directory already exists → skip simulation. Pass
`--force` to re-run and overwrite existing results.

---

## 20. Parameter Files

### 20.1 Values (v0.1)

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

### 20.2 Priors

Priors are declared with `~` syntax directly on parameters in the `.camdl` file
— they are beliefs about parameters and belong with the declaration:

```camdl
parameters {
    beta  : rate in [0.01, 2.0] ~ log_normal(mu = -1.0, sigma = 0.5)
    gamma : rate in [0.05, 1.0] ~ half_normal(sigma = 0.3)
    rho   : probability in [0.001, 1.0] ~ beta(alpha = 2.0, beta = 5.0)
    N0    : count in [100, 1_000_000]  # no prior — must be supplied via
                                       # --params, --scenario, or [fixed] in fit.toml
}
```

The `~` reads "distributed as" and is always optional. Parameters without a
prior can still be fixed at a known value via `--params` files.

**Supported distributions**:

| Distribution       | Syntax                                       | Parameters              |
| ------------------ | -------------------------------------------- | ----------------------- |
| `uniform`          | `~ uniform(lower = L, upper = U)`            | bounds                  |
| `normal`           | `~ normal(mu = M, sigma = S)`                | mean, sd (natural)      |
| `log_normal`       | `~ log_normal(mu = M, sigma = S)`            | log-scale mu, sigma     |
| `half_normal`      | `~ half_normal(sigma = S)`                   | sd of underlying normal |
| `beta`             | `~ beta(alpha = A, beta = B)`                | shape parameters        |
| `gamma`            | `~ gamma(shape = K, rate = R)`               | shape, rate (NOT scale) |
| `exponential`      | `~ exponential(rate = R)`                    | rate = 1/mean           |
| `log_uniform`      | `~ log_uniform(lower = L, upper = U)`        | bounds, `L, U > 0`      |
| `truncated_normal` | `~ truncated_normal(mean = M, sd = S)`       | mean, sd; bounds from `in [..]` |

All arguments are keyword (named), never positional. All arguments must be
compile-time constants.

**Parameterization conventions** (these are load-bearing):

- `log_normal(mu, sigma)`: parameters are on the **log scale**.
  `log(X) ~ Normal(mu, sigma)`. Median of X is `exp(mu)`.
- `half_normal(sigma)`: sigma is the SD of the underlying (unfolded) normal.
- `gamma(shape, rate)`: rate parameterization (`E[X] = shape/rate`).
- `log_uniform(lower, upper)`: **uniform on the log scale** —
  `log(X) ~ Uniform(log lower, log upper)`, so every order of magnitude in
  `[lower, upper]` is equally likely. The honest weakly-informative choice for a
  scale parameter known only to within orders of magnitude (e.g. a coupling
  rate). Requires `lower, upper > 0`. Use the `Log` transform (the default for
  `rate`/`positive` parameters).
- `truncated_normal(mean, sd)`: a `normal(mean, sd)` truncated to the
  parameter's declared `in [lo, hi]` range — the bounds are the truncation,
  with no second place to disagree. A parameter with a `truncated_normal` prior
  **must** declare `in [lo, hi]`. Exact and warning-free, unlike a plain
  `normal(...)` whose out-of-bounds mass is rejected at draw time.

Priors in the model are the primary source; `fit.toml [estimate]` priors
override them for sensitivity analysis. See the run spec §12 for the full
precedence chain.

### 20.3 Views (v0.2+)

```toml
# view.toml — implements V from the parameter grammar
[view]
free = ["beta", "gamma", "rho", "I0"]
```

Free parameters are varied by the inference engine; all other parameters are
held fixed at their values from `params.toml`. Views are only relevant for
`camdl fit` (v0.2+) — they have no effect on forward simulation.

### 20.4 Relationship to the Parameter Grammar

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

## 21. CLI

The unified `camdl` command routes to two backends: **`camdlc`** (OCaml
compiler) for compilation/inspection and **`camdl`** (Rust) for simulation,
experiments, and inference. All commands accept `.camdl` files directly
(auto-compiled via `camdlc`).

### 21.1 Compilation and Inspection

```bash
camdl dev compile MODEL.camdl          # compile to IR JSON (stdout)
camdl check   MODEL.camdl              # validate structure (no output)
camdl inspect MODEL.camdl [OPTIONS]    # inspect compartments, transitions, etc.
```

### 21.2 Simulation

```bash
camdl simulate MODEL --params P.toml --seed 42 [OPTIONS]

Options:
  --backend    gillespie|chain_binomial|ode  (default: chain_binomial)
  --dt         DT         step size for chain_binomial / ode
  --seed       N          RNG seed (default: 1)
  --scenario   NAME       select a named scenario
  --enable     NAME       enable an intervention (ad-hoc; mutually exclusive with --scenario)
  --disable    NAME       disable an intervention
  --param      NAME=VALUE override a parameter value
  --param-vec  PREFIX=FILE override indexed params from a keyed TSV
  --params     FILE.toml  load parameter values (repeatable, later overrides earlier)
```

By default `simulate` writes the trajectory to a content-addressed store leaf
under `./results` (read it back with `camdl cat <id>`); pass `--stdout` to stream
the TSV to stdout instead, or `-o FILE` to also write a loose TSV mirror. The TSV
columns are `t`, one column per compartment, and `flow_<name>` per transition.

### 21.3 Expression Evaluation

```bash
camdl dev eval MODEL --params P.toml --expr "school,seas,R0" --from 0 --to 365 --every 1
camdl dev eval MODEL --params P.toml --expr "school" --at 0,100,200,300
```

Evaluates time-dependent expressions (forcing functions, parameters, math
expressions) at a time grid without simulation. Expressions that reference
compartment state produce an error.

### 21.4 Batch simulation

```bash
camdl batch run     BATCH.toml [--parallel N] [--force] [--dry-run]
camdl batch status  BATCH.toml
camdl list              [RESULTS_DIR]          # browse cached runs
camdl show  <short-hash>
camdl cat   <short-hash> [--stream NAME]
```

Batch parameter sweeps, scenario comparisons, and posterior-predictive checks.
Sensitivity analysis (Sobol indices) is out of scope for the CLI; compute it
with R's `sensitivity` package or Python's `SALib` on the batch output. See the
Run Specification (`camdl-run-spec.md` §5) for details.

### 21.5 Inference

```bash
# Particle filter — log-likelihood estimation
camdl pfilter MODEL --params P.toml --data cases.tsv \
    --particles 5000 --dt 1 --seed 42 --trace diag.tsv

# Iterated filtering (MLE) is run as a one-stage fit. Write a fit.toml with a
# single `algorithm = "if2"` stage and run it through `camdl fit run`:
#
#   [model]
#   camdl = "MODEL"
#
#   [data.observations]
#   cases = "cases.tsv"
#
#   [estimate]
#   R0    = { bounds = [0.1, 10.0], start = 5.0 }
#   sigma = { bounds = [0.0, 1.0],  start = 0.01 }
#   gamma = { bounds = [0.0, 1.0],  start = 0.01 }
#
#   [fixed]
#   N0 = 1000
#   mu = 0.0
#   k  = 10
#
#   [stages.fit]
#   algorithm  = "if2"
#   backend    = "chain_binomial"
#   chains     = 4
#   particles  = 2000
#   iterations = 100
#   cooling    = 0.95
#
camdl fit run fit.toml --seed 42

# Profile likelihood — parameter identifiability. The swept parameter and its
# grid are given by `--sweep "PARAM=lin(min,max,n)"` (repeat for 2D+).
camdl profile MODEL --init from_params --params P.toml --data cases.tsv \
    --sweep "R0=lin(0.5,5,20)" \
    --rw-sd "sigma=0.01,gamma=0.01" \
    --particles 500 --iterations 30 --starts 3 --parallel 8

# 2D profile — repeat --sweep
camdl profile MODEL \
    --sweep "alpha=lin(0.85,0.95,3)" --sweep "gamma=lin(0.06,0.10,3)" ...
```

The projection and likelihood for each data stream come from the model's
`observations { ... }` block (§12); inference commands do not take a `--flow` /
`--obs-model` projection override. When a single positional `--data FILE` is
bound and the model declares more than one observation stream, pass `--obs NAME`
to say which stream (or indexed family) the file is. The legacy `--flow` /
`--obs-model` projection-override flags were removed in the 2026-05-25 CLI UX
revision.

**`--rw-sd`** (`camdl profile`): Perturbation scale per parameter. Three modes:

- Explicit: `--rw-sd "R0=5,sigma=0.01"` — the list IS the partition. Parameters
  not listed are held fixed. No `--fixed` needed.
- Auto: `--rw-sd auto` — heuristic from parameter bounds (`(hi-lo)/6` on
  transformed scale). Use `--fixed NAME=VALUE` (repeatable) or
  `--fixed-file <toml>` to exclude and value-pin specific params.
- Mixed: `--rw-sd "R0=5,sigma=auto"` — explicit where you know, auto where you
  don't.

In a fit, an `algorithm = "if2"` stage derives its perturbation scale from each
parameter's declared `bounds` — there is no per-parameter `rw_sd` knob on the
stage. Cooling is pomp's cf50 convention (halfway-SD fraction); see
`docs/methods/cooling.md`.

**Regimes (scout / refine / validate)**: the scout → refine → validate ladder is
a sequence of `[stages.X] algorithm = "if2"` blocks in a `fit.toml`, not a CLI
preset. A scout is a fast, mildly-cooled stage for basin exploration (e.g.
`chains = 8`, `particles = 500`, `iterations = 30`, `cooling = 0.70`); a refine
sharpens onto the scout's mode with more particles and aggressive cooling (e.g.
`chains = 4`, `particles = 1000`, `iterations = 50`, `cooling = 0.05`,
`init_mle = "scout"`); a validate stage is a final high-particle polish. Each
stage sets these knobs explicitly; a later stage warm-starts from an earlier
one's MLE via `init_mle = "<stage>"`. A scout-convergence gate
(`docs/methods/cooling.md`) guards the transition.

**Initial value parameters (IVP)**: parameters that set the initial compartment
state (e.g. `S0`, `I0`) are declared on the model and estimated like any other
parameter — list them under `[estimate]` in the `fit.toml`. The fit perturbs /
draws their initial values as part of the inference; there is no separate CLI
flag to nominate them.

**`--fixed NAME=VALUE`**: Pin `NAME` at `VALUE` (repeatable) and remove it from
the inference `[estimate]` set if present. The universal value-setter, available
on `camdl profile` and `camdl survey`; in a `fit.toml` the equivalent is the
`[fixed] NAME = VALUE` block read by `camdl fit run`.

**Pinning many params (the replacement for the removed name-only `--fixed`).**

The pre-2026-05-25 surface accepted a name-only comma list: `--fixed "N0,mu,k"`
meaning "freeze these three at their model defaults." That form is **removed**.
The two replacements are:

- `--fixed NAME=VALUE` (repeated): explicit values, one per flag. Preferred for
  the small case (≤ 3 names).
- `--fixed-file <toml>`: a flat params TOML — top-level keys are parameter
  names, values are the pin values; repeatable, later files override earlier
  ones. Preferred for many-params vignettes (extract the values once, commit the
  TOML, point the invocation at it).
- For the original "pin at the model default" intent — i.e. when the user didn't
  want to type any values at all — the equivalent under the new surface is to
  simply **not list the parameter in `--fixed`/`--fixed-file`**. The model
  default flows through the precedence chain unchanged. The previous spelling
  expressed "freeze at default" as an explicit gesture; the new spelling
  expresses it as the absence of one. See `docs/camdl-run-spec.md §1.3` for the
  full precedence chain and
  `docs/dev/proposals/archive/post-alpha/2026-05-25-cli-init-and-params-ux.md` §"`--fixed`
  semantics, defined once" for the rationale.

### 21.6 Fit Workflow

```bash
camdl fit run     fit.toml [--stage NAME] [--seed N] [--force] [--sweep "PARAM=V1,V2,..."]
camdl fit summary results/fits/<dir>/
camdl fit table   results/fits/
```

Driven by `fit.toml` with `[estimate]`, `[fixed]`, `[data]`, and one or more
`[stages.NAME]` blocks. Stages are named by the user (by convention `scout`,
`refine`, `validate`) and chain via the `init = "from_mle"` +
`init_mle = "<prior-stage>"` pair on each stage. `--stage NAME` runs a single
stage; `--sweep` takes a Cartesian product over parameter grids and, when a cell
fails the convergence gate, records the failure in `sweep_failures.tsv` and
continues rather than halting. See `docs/camdl-inference-spec.md`.

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

Columns:
`time ll_increment ESS obs_mean obs_q05 obs_q50 obs_q95
state_mean state_q05 state_q50 state_q95 observed`.
`obs_*` includes observation noise; `state_*` is process uncertainty only.

### 21.7 Data Utilities

```bash
# Split a TSV at a time threshold for train/holdout validation
camdl data split data/cases.tsv --at-time 5474
# → data/cases_train.tsv, data/cases_holdout.tsv

# Explicit output paths
camdl data split data/cases.tsv --at-time 5474 \
    --train data/train.tsv --holdout data/holdout.tsv
```

### 21.8 Particle State Export

```bash
camdl pfilter MODEL --params P.toml --data train.tsv \
    --particles 5000 --save-final-state final_particles.tsv
```

---

## 22. Worked Examples

These examples progress from trivial to complex, showing how primitives compose.
Each shows the DSL source and key points about what the compiler generates.

### 22.1 Bare SIR (Simplest Possible Model)

```camdl
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

### 22.2 SIR with Demography (Explicit Transitions)

```camdl
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

### 22.3 SEIR with Age Mixing (Introducing Stratification)

Age-structured transmission is written with the explicit indexed form (§10): one
`infection` transition per age class, with a contact-matrix-weighted sum over the
infectious classes.

```camdl
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

The `sum(b in age, C_age[a,b] * I[b] / N_local[b])` term is the
contact-matrix-weighted force of infection on age class `a`; progression and
recovery are replicated within each stratum by the `[a in age]` binder. (An
earlier `coupling[age = C_age]` shorthand that auto-generated this sum was
removed — see §10.)

### 22.4 STI with Directed Transmission (Off-Diagonal Matrix)

```camdl
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

### 22.5 Cholera with Environmental Reservoir (Real Compartment + ODE)

```camdl
time_unit = 'days

compartments {
  S, I, R,
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

### 22.6 Five-Age-Group Model with Consecutive Aging

<!-- camdl-doctest-data: data/polymod_5x5.tsv
age_i	age_j	contact
age_0_5	age_0_5	1.5
age_0_5	age_5_15	1.5
age_0_5	age_15_50	1.5
age_0_5	age_50_65	1.5
age_0_5	age_65p	1.5
age_5_15	age_0_5	1.5
age_5_15	age_5_15	1.5
age_5_15	age_15_50	1.5
age_5_15	age_50_65	1.5
age_5_15	age_65p	1.5
age_15_50	age_0_5	1.5
age_15_50	age_5_15	1.5
age_15_50	age_15_50	1.5
age_15_50	age_50_65	1.5
age_15_50	age_65p	1.5
age_50_65	age_0_5	1.5
age_50_65	age_5_15	1.5
age_50_65	age_15_50	1.5
age_50_65	age_50_65	1.5
age_50_65	age_65p	1.5
age_65p	age_0_5	1.5
age_65p	age_5_15	1.5
age_65p	age_15_50	1.5
age_65p	age_50_65	1.5
age_65p	age_65p	1.5
-->

```camdl
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

### 22.7 Erlang Sub-Staging (Non-Exponential Waiting Times)

```camdl
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

## 23. Full Example: Spatial Age-Structured SEIR

<!-- camdl-doctest-data: data/lga_pop.tsv
patch	pop
p1	1200000
p2	850000
-->

<!-- camdl-doctest-data: data/lga_dist.tsv
patch	patch	distance
p1	p2	140.0
p2	p1	140.0
-->

```camdl
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

# Gravity kernel. The diagonal is excluded where the kernel is USED — by a
# `where` predicate, which is decidable at compile time — rather than by an
# `if` inside the binding: `if i == j` would compare index variables in an
# expression, where they have already been substituted for level names.
let mig[i in patch, j in patch] = theta * pop[j] / (distance[i,j] ^ 2)

## ── Functions ──────────────────────────────────────────

forcing {
  seasonal : sinusoidal 'ratio {
    amplitude = alpha
    period    = 365.25 'days
    phase     = phi_season
    baseline  = 1.0
  }
}

## ── Transitions ────────────────────────────────────────

transitions {
  # Infection: age mixing, spatial coupling via gravity kernel
  infection[a in age, p in patch] : S[a,p] --> E[a,p]
    @ beta * seasonal * S[a,p]
      * sum(b in age, q in patch where q != p,
          C_age[a,b] * mig[p,q] * I[b,q] / N_local[b,q])

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

## ── Observations ───────────────────────────────────────

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    # `infection` is stratified over age × patch, so the pooling is stated
    # rather than left implicit — a bare `incidence(infection)` here is E280
    # (§12.1). One national column, one reporting rate.
    projected     = sum(a in age, p in patch, incidence(infection[a, p]))
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

## ── Init ───────────────────────────────────────────────
# Minimal test initialization — single patch seeding.
# For a full 774-patch model, use per-patch init from a table.

init {
  S[child, p1]  = 100000
  S[adult, p1]  = 200000
  I[child, p1]  = I0
  # All other compartments/strata default to 0
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

## 24. Compilation Pipeline

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

### 24.1 File-Level Grammar

A `.camdl` file is a sequence of declarations. Order does not matter — all
declarations are collected first, then resolved (forward references are valid).

```camdl
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
  | events_block                      # events { ... }
  | ode_block                         # ode { ... }
  | output_block                      # output { ... }
  | quantities_block                  # quantities { ... }
  | timepoints_block                  # timepoints { ... }
  | init_block                        # init { ... }
  | simulate_block                    # simulate { ... }
  | scenarios_block                   # scenarios { ... }
  | balance_block                     # balance { ... }
  | stratify_decl                     # stratify(by = ..., ...)
  | let_binding                       # let NAME = EXPR
```

**Mandatory** for a runnable model: `compartments`, at least one of
`transitions` or `ode` (integer compartments evolve via `transitions`, `real`
compartments via `ode`; a model may use either or both), `init`, `simulate`, and
`time_unit`. Everything else is optional.

**Mandatory** for `camdl check` (validation only): `compartments`. A param-free
model checks cleanly and a pure-`ode` model (no `transitions` block) checks and
compiles; no `parameters`, `simulate`, or `init` is required for validation.

**Expander** (OCaml): indexed transitions → flat IR transitions,
`c in compartments` → per-compartment transitions, `consecutive` → adjacent pair
transitions, `sum(... where P, ...)` and transition `where` → compile-time
filtering, let bindings → inlined expressions, `ode {}` → IR ODE equations, unit
normalization.

**Validator**: compartment arity checking, table dimension checking, index
variable scoping, parameter reference resolution, dimensional analysis.

**Serializer**: expanded IR → compact one-element-per-line JSON (a `--pretty`
variant exists for inspection).

**Runtime** (Rust): deserializes IR, evaluates propensities, simulates, writes
output. Knows nothing about the DSL — sees only flat compartments, transitions,
and expression ASTs.

---

## 25. Expansion Rules (DSL → IR Mapping)

Every DSL construct compiles to specific IR structures. This section documents
the mapping for each construct — the contract between the OCaml frontend and the
Rust backend.

### 25.1 Let Bindings

```camdl
# DSL:
let N = S + E + I + R

# IR: everywhere N appears, inline the expression tree:
BinOp(Add, BinOp(Add, BinOp(Add, Pop("S"), Pop("E")), Pop("I")), Pop("R"))
```

After stratification, bare `S` in the let body becomes
`PopSum(["S_child", "S_adult"])`. N is always the global total.

### 25.2 Indexed Transitions

```camdl
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

### 25.3 Inflows

```camdl
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

### 25.4 Projections

```camdl
# DSL (UNSTRATIFIED `infection` only — see §12.1: a bare, un-indexed
# incidence() over a STRATIFIED family on an un-indexed stream is E280):
incidence(infection)

# IR:
CumulativeFlow("infection")

# DSL: pooling a stratified family, stated explicitly
sum(a in age, incidence(infection[a]))

# IR:
CumulativeFlowSum(["infection_child", "infection_adult"])

# DSL: adding distinct flows into one reported column (two routes into the
# same notification series). Addition and the family sum are the same object —
# a set of flows accumulated over the stream's interval — so they compose and
# flatten into one list:
incidence(infection[child]) + incidence(infection[adult])
sum(a in age, incidence(infection[a])) + incidence(importation)

# IR:
CumulativeFlowSum(["infection_child", "infection_adult"])
CumulativeFlowSum(["infection_child", "infection_adult", "importation"])

# A projection may ADD incidence terms; it may not weight, subtract, or mix
# them with an instant state read (E341). A per-stream reporting rate goes in
# the LIKELIHOOD (`cases ~ poisson(rate = rho * projected)`); a PER-STRATUM
# weight inside the projection is not yet supported.

# DSL:
incidence(infection[child])    # indexed: specific stratum

# IR:
CumulativeFlow("infection_child")

# DSL:
prevalence(R)                  # bare: global total

# IR:
CurrentPopSum(["R_child", "R_adult"])
```

### 25.5 Interventions

```camdl
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

### 25.6 Stratified Transmission (Indexed Sum → IR)

```camdl
# DSL: explicit indexed transmission (§10)
infection[a in age] : S[a] --> E[a]
  @ beta * S[a] * sum(b in age,
      C_age[a,b] * I[b] / sum(c in compartments, c[b]))
# with age = [child, adult]

# Expands to IR:
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

The per-stratum denominator `sum(c in compartments, c[b])` becomes `PopSum` of
all compartments in stratum `b`.

### 25.7 Consecutive Pairs

```camdl
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

### 25.8 Guard Clauses (`where`)

Guards are evaluated at compile time. The compiler instantiates all index
combinations, evaluates the guard, and **omits** transitions where the guard is
false. The IR has no concept of guards.

```camdl
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

### 25.9 Compartment Iteration (`c in compartments`)

The compiler expands `c in compartments` by substituting each integer
compartment name. When a compartment has more dimensions than the index
signature provides, the compiler **iterates over the omitted dimensions** to
satisfy the stoichiometry rule (§5.1).

```camdl
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

### 25.10 Interventions (All Dimensions)

Interventions on stratified compartments expand over **all** dimensions:

```camdl
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

Endpoints pair only when both are declared with the same dimensions in the same
order; anything else is **E237**. `count =` on this bare form is **E238**, and a
bare endpoint inside an indexed family is **E239** (all three: §13.1).

---

## 26. Errors and Validation

The compiler produces clear, domain-specific error messages. Errors are caught
at compile time, not simulation time.

### 26.0 Diagnostic Codes

Diagnostics carry a numeric code for programmatic consumption (e.g.,
`--json-errors` mode). The codes below are a representative selection — the
families (not every individual code) are: `E001` (parse/lexer), `E1xx`
(names/indices/reserved-ids), `E2xx` (declaration, expansion, schedule, and
observation-surface checks), `E3xx` (dimensional analysis), `E4xx`
(forcing/table), `E6xx` (validator), and `W1xx`/`W2xx`/`W3xx` (warnings). The
authoritative list is whatever `ocaml/lib/compiler/` and `ocaml/lib/ir/` emit.

| Code | Kind    | Description                                                                                |
| ---- | ------- | ----------------------------------------------------------------------------------------- |
| E001 | Error   | Parse / lexer syntax error (including using a keyword as a name)                           |
| E100 | Error   | Reserved name used as a declaration; unknown index value; undeclared name/function         |
| E200 | Error   | Undeclared compartment or parameter referenced in expression                              |
| E202 | Error   | Table arity / shape mismatch (e.g. `table '%s' expects %d indices`)                        |
| E228 | Error   | `time_unit` is not a duration unit (`'per_day` / `'count` / `'ratio`) — see §2               |
| E265 | Error   | Intervention `set`/`add` targets a name that is not one expanded compartment               |
| E272 | Error   | Removed observation cadence form — use `emit_schedule = …` (§12.4)                         |
| E278 | Error   | Duplicate declaration (a name declared more than once / in multiple namespaces)            |
| E300 | Error   | Transition rate has the wrong dimension                                                    |
| E302 | Error   | Addition/subtraction of mismatched dimensions                                             |
| E303 | Error   | Conflicting dimensions for a parameter across transitions                                 |
| W103 | Warning | Let binding name shadows a stratum value in some dimension                                |

Diagnostics can be emitted as structured JSON by passing `--json-errors` to
`camdlc`:

```bash
camdlc check model.camdl --json-errors 2>errors.json
```

### 26.1 Dimension Errors

```camdl
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

### 26.2 Unbound Variables

```camdl
infection[a in age] : S[a] --> E[a]
  @ beta * S[a] * I[a, s] / N
# ERROR at line 45: 's' is used in I[a, s] but is not bound.
#   I has dimensions [age, sex]. Bind 's' with 'sum(s in sex, ...)'
#   or add 's in sex' to the transition index.
```

### 26.3 Partial Stratification Stoichiometry

```camdl
dimensions { immunity = [natural, vaccine] }
stratify(by = immunity, only = [R])

recovery[a in age] : I[a] --> R[a]  @ gamma * I[a]
# ERROR at line 55: R has dimensions [age, immunity] but destination
#   R[a] only specifies [age]. All dimensions of a destination
#   compartment must be specified in stoichiometry.
#   Did you mean: R[a, natural] or R[a, vaccine]?
```

### 26.4 Dimension Does Not Exist

```camdl
recovery[a in age, r in habitat] : I[a, r] --> R[a, r]  @ gamma * I[a, r]
# ERROR at line 50: 'habitat' is not a declared dimension.
#   Declared dimensions: age, sex, patch.
```

### 26.5 Compartment Doesn't Have Dimension

```camdl
dimensions { immunity = [natural, vaccine] }
stratify(by = immunity, only = [R])

waning[a in age] : I[a, natural] --> S[a]  @ wane * I[a, natural]
# ERROR at line 55: I does not have dimension 'immunity'.
#   I has dimensions: [age]. Only R has dimension 'immunity'.
#   Did you mean R[a, natural]?
```

### 26.6 Unit Errors

```camdl
transitions {
  recovery : I --> R  @ gamma + I
  # where gamma : rate ('per_day) and I : count
}
# ERROR at line 33: cannot add rate (1/time) and count (dimension P).
#   Did you mean 'gamma * I'?
```

### 26.7 Parameter Domain Errors

Checked when parameter values are supplied (not at model compile time):

```
# params.toml: rho = 1.5
# ERROR: parameter 'rho' is declared as probability (∈ [0, 1])
#   but supplied value is 1.5.
```

### 26.8 Scenario Validation

```camdl
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

### 26.9 Self-Loop Detection

A generated self-loop is a **hard error** (`E310`), not a warning. When an
expanded transition's source and destination are the same compartment, its net
stoichiometry cancels to empty, and the compiler rejects it rather than emitting
a no-op transition:

```camdl
migrate[c in compartments, src in patch, dst in patch]
  : c[src] --> c[dst]  @ mig[dst, src] * c[src]
# ERROR E310: transition 'migrate' has no net effect: sources and
#   destinations cancel (the diagonal src == dst members are self-loops).
#   Add 'where src != dst' to drop the diagonal, or ensure mig[i,i] = 0.
```

Add the guard to filter the diagonal:

```camdl
migrate[c in compartments, src in patch, dst in patch]
  : c[src] --> c[dst]  @ mig[dst, src] * c[src]  where src != dst
```

### 26.10 Name Resolution

A user-defined name lives in exactly one namespace. The same name declared in
two of {compartments, parameters, let bindings, forcing, tables} is a
**duplicate-declaration** error (`E278`), so there is never a legal
cross-namespace collision for a resolution order to arbitrate — the compiler
rejects the ambiguity rather than silently preferring one namespace. What is
reported:

- **Shadowing reserved identifiers**: `t_start`, `t_end`, `compartments`, etc.
  (`E100`).
- **Duplicate declarations** (`E278`): two parameters named `beta`, two
  compartments named `S`, or the same name in two of the namespaces above (e.g.
  a parameter and a compartment both named `N`).

### 26.11 Compiler Reporting

For every model, `camdl check` reports a size summary. The real output uses the
model name as a bold header, lowercase field labels, a `→` between base and
expanded counts, and a "(+ N filtered by where)" suffix on the transitions line.
The exact glyphs and colouring are terminal-dependent; illustratively:

```
seir_age_seasonal

  compartments   5 base × 2 age × 774 patch = 7740 expanded
  transitions    8 base → 47,892 expanded (+ 0 filtered by where)
  parameters     13 declared
  ...
```

This gives the user a quick sanity check on model size before simulation. (For
the full per-parameter listing use `camdlc inspect --parameters`.)

---

## 27. Primitive Summary

The language is built on these composable primitives:

```camdl
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
sin(x), cos(x), tanh(x)             trigonometric / hyperbolic (unary)
mod(a, b)                            Euclidean remainder (binary)
min(a, b), max(a, b)                element-wise min / max (binary)

# Time
t                                    current simulation time

# Forcing (time-dependent functions) — the tier-3 unit literal 'UNIT is required (§7; E001 if omitted)
forcing { NAME : sinusoidal 'UNIT { ... } }     smooth seasonal
forcing { NAME : periodic 'UNIT { ... } }       repeating step (values or on = [lo:hi, ...])
forcing { NAME : piecewise 'UNIT { ... } }      non-repeating step
forcing { NAME : interpolated 'UNIT { ... } }   data-driven (linear or spline)

# Reserved identifiers
t, t_start, t_end, dt, pi, e                # genuinely reserved (E100)
compartments, sum, consecutive             # keywords (E001 if used as a name)
```
