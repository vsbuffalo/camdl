# The camdl Language Specification

**Version:** 0.3-draft **Date:** 2026-07-17

_camdl (Compartmental Model Description Language) is a domain-specific language
for specifying stochastic compartmental models. A `.camdl` file defines model
structure. Parameter values, inference configuration, and scenario selection are
supplied externally._

---

## 1. Design Principles

This section states the rules that hold across the whole language. Each is
specified in detail in the section it links to.

**Primitives and sugar.** The language is a small set of primitives. A
convenience form is specified by the primitives it expands to (§25), and the
expanded form can be written by hand.

**Rates are total propensities.** Every rate expression after `@` is the total
rate of the transition in the whole source population, in individuals per unit
time. The compiler never multiplies a rate by a population: a per-capita rate is
written with its population factor, `gamma * I`. Dimensional analysis refuses a
rate that is not a population per unit time (**E300**, §2.2.1).

**One model file, values supplied around it.** A `.camdl` file is the same file
for simulation, calibration, scenario comparison and forecasting; what changes
between analyses is the configuration layered on top. A file takes one of two
shapes:

- a **structural skeleton**, which declares parameter names, kinds and
  dimensions only, and is given values by a `--params` TOML file, CLI flags or
  an inference method;
- a **self-contained model**, which adds `~` priors (§20.2), an `init { }`
  block (§15) and a `baseline` scenario whose `set = { … }` gives every
  parameter a value (§17), so that the file runs end to end on its own. The
  examples shipped with camdl take this shape.

The `parameters { }` block never holds a value: it declares names, kinds,
dimensions, bounds and priors (§4). A value comes from a `--params` file or a
fit's `[fixed]` table, from a CLI flag (`--param`, `--fixed`), or, as the only
in-file form, from a scenario's `set = { … }`. The order in which these sources
override each other is specified in the run spec (§1.3). It is not "external
beats in-file": a selected scenario's `set` and `scale` override the
`--params` base and any sweep point, `--param` and `--fixed` override every
parameter the scenario does not set or scale, and a `--param` or `--fixed` on a
parameter the scenario sets or scales is refused. The seed is always a CLI
argument.

A run's identity is factored into levels (§19.1). The `model` level hashes the
structure (compartments, transitions, observations, ODE equations, tables,
`init`, `time_unit`); parameter values, the backend and `dt`, and the scenario
delta are hashed in their own levels. Two runs of one structural model
therefore share the `model` hash whether the values were written in a scenario
or supplied from outside, and a change of value re-keys only the level that
holds it.

**Compile-time checks.** Index dimensions, table shapes, compartment arities,
parameter domains and dimensions are checked when the model is compiled, with
a diagnostic code for each failure (§26). A **named** index
(`I[age = child]`) is resolved by the dimension it names, and a name that is not
a dimension of the indexed family is **E332**. A **positional** index
(`I[child]`) is checked against the levels of the dimension in that slot, and a
level that is not one of them names no cell and is refused: **E100** on a
compartment, **E504** on a parameter.

The compiler does not record which dimension an index _variable_ was bound to.
With `age = [low, high]` and `risk = [low, high]`, `I[r, a]` with `a` bound to
`age` and `r` to `risk` names an existing cell and compiles without a
diagnostic. Positional indexing is therefore checked only while the dimensions'
level names are disjoint; where two dimensions could share a level name, use
named indexing (§12.1).

**No auto-localization.** In model dynamics a bare compartment name of a
stratified compartment is its total over all strata: `S` is all susceptibles
and `S[child]` the susceptible children. The compiler never chooses a stratum.
Stratified dynamics are written with explicit indexed transitions and sums
(§9.2, §10). In an observation's projection an unindexed name over a population
stratum is refused instead (**E280**, §12.1), so that pooling across strata is
written out.

### 1.1 Syntax Conventions

```camdl
:    structural definition (what something IS)
=    value binding (what something EQUALS)
@    rate expression (how fast, always total propensity)
-->  flow direction (Unicode → is an exact alias)
#    comment
#'   doc comment (attaches to the declaration below it; at the top
     of the file, to the model as a whole)
{ }  block grouping
[ ]  index access and list literals
( )  function arguments
'    unit literal prefix ('days, 'years)
```

**Arrow.** The flow arrow is `-->`. The Unicode rightwards arrow `→` (U+2192) is
an exact alias, so a transition pasted from prose, or typed in an editor that
turns `->` into `→`, compiles as written. The arrow separates sources from
destinations and has no other meaning; the two spellings produce identical IR.

```camdl
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
  N0    : count
  I0    : count
}
let N = S + I + R
transitions {
  infection : S → I @ beta * S * I / N
  recovery  : I → R @ gamma * I
}
init {
  S = N0 - I0
  I = I0
}
simulate { from = 0 'days  to = 120 'days }
```

#### Doc comments (`#'`)

A line starting `#'` documents the declaration immediately below it. A plain `#`
comment is read by no tool; `#'` prose is carried through compilation and shown
by `camdlc inspect`, `camdlc render` and the parameter legend of
`camdl fit summary`.

A doc comment attaches to a declaration, never to a block keyword. Inside a
block it documents the member below it:

```camdl
compartments {
  #' fully susceptible
  S,
  #' confirmed and still transmitting — the infectious dwell ends here
  C
}
```

The declarations that take a doc comment are `compartments` members,
`parameters` members, `dimensions` entries, `transitions`, `observations`
streams, `quantities`, and top-level `let` bindings. A `#'` line above a block
keyword (`compartments {`, `transitions {`), or with no declaration below it, is
a syntax error, **E001**.

**The model's doc comment.** A `#'` block at the top of the file, where every
line above it is blank or a comment, documents the model itself:

```camdl
#' National SEIR with a facility-death delay: cases and deaths come from one
#' confirmation flow, deaths lagged through an isolation compartment.
#' @base bvd_national_twocfr.camdl
#' @adds nothing
#' @changes f_cfr_unret becomes free with a beta(2,2) prior

time_unit = 'days
```

The rule is positional, so in a file whose first declaration is a top-level
`let`, a `#'` block above it documents the model and the `let` has no doc
comment; to document that `let`, place it after another declaration.

**Tags.** On a declaration, two tags are recognized, and any other `@tag` line
is **E111**:

```camdl
#' basic reproduction number
#' @symbol R_0
#' @ref Anderson & May 1991, ch. 6
R0 : positive in [0.5, 20]
```

`@symbol` sets the symbol `camdlc render` prints for the name in its LaTeX and
JSON output, wherever the name is typeset, so a model identifier spelled for
code (`Conf`) can be printed as the paper spells it (`C`). `@ref` records a
citation or a case definition. Both are optional, and either may appear without
prose. In the model's doc comment the vocabulary is open: `@symbol` and `@ref`
behave as on a declaration, and any other `@tag` line (such as `@base`, `@adds`
and `@changes` above) is kept verbatim as prose. The compiler checks nothing
about these lines, including whether a file they name exists.

**Doc comments and run identity.** A model's doc comment is stored outside the
model's IR and is not part of run identity, so it can be edited without
re-keying any run. A `description = "…"` declaration is part of the model's IR
and of its `model` hash (§19.1), so editing it re-keys every run and fit of the
model (§16.4).

**Doc comments on `let` bindings.** A doc comment on a `let` states what a
derived quantity is, which its arithmetic does not: `S[p] + E[p] + I[p] + C[p]`
may be the total population or the force-of-infection denominator.

```camdl
#' total population per patch — the FOI denominator
let N[p in patch] = S[p] + E[p] + I[p] + C[p]
```

A **typed** `let` with a constant body is a **model constant**:
`let omega : rate = 0.01` compiles to an entry of the IR's `constants` section,
with its value and its declared kind, and is not a parameter. It holds its value
in every run. `--fixed`, `[fixed]`, `[estimate]`, `--sweep` and a scenario's
`set` cannot name it; one that tries is refused as an unknown parameter, with a
hint to declare it in `parameters { }` to set it from outside.
`camdl fit summary --parameters` lists the constants under their own
`constants` heading after the parameter legend, each with its value and its
prose. Every other `let` is inlined into the expressions that use it; its prose
is shown by `camdlc inspect --let <name>`. `@symbol` applies to both.

Related sections: §8 (let bindings), §8.4 (typed let bindings), §16.4
(quantities), §24.1 (the file-level grammar).

---

## 2. Time Unit and Dimensional Types

`time_unit` sets the unit of model time. Every rate and duration in the model is
converted to it when the model is compiled.

```camdl
time_unit = 'days
```

`time_unit` must be a duration unit: `'days`, `'weeks`, `'months` or `'years`.
A rate (`'per_day`), `'count` or `'ratio` is not a length of time and is
**E228**. `'months` and `'years` cannot be combined with `origin = date(…)`
(**E320**), because a calendar month is not a constant number of days (§2.1).

### 2.1 Unit Literals

A unit literal is a number followed by a unit name with a `'` prefix:

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

The units are `'days`, `'weeks`, `'months` and `'years` (time, T);
`'per_day`, `'per_week`, `'per_month` and `'per_year` (T⁻¹); `'count`
(population, P); and `'ratio` (dimensionless). `'count` and `'ratio` give a
dimension to a value outside the time-or-rate axis, on table cells (§2.5) and
forcing declarations (§7). A parameter's dimension is written with a `dim`
clause, never a unit literal (§4.1.1).

The conversions are 1 'weeks = 7 'days, 1 'years = 365.2425 'days (the mean
year of the proleptic Gregorian calendar), and 1 'months = 365.2425/12 'days ≈
30.4369 'days.

#### Exact and calendar durations

A duration is either **exact** or **calendar**:

| Unit literal        | Kind     | Length                                           |
| ------------------- | -------- | ------------------------------------------------ |
| `'days`, `'weeks`   | Exact    | a constant number of days (1, 7)                 |
| `'months`, `'years` | Calendar | the calendar's mean month or year (≈ 30.4369, ≈ 365.2425 days) |

The kind is a refinement of the time dimension T, with Exact a subtype of
Calendar. It propagates through arithmetic by least upper bound: `Exact ± Exact`
is Exact, and an expression containing a `'months` or `'years` literal is
Calendar.

The distinction applies only in **anchored mode**, when the model declares
`origin = date(…)` (§2.3). A calendar month has no fixed length relative to a
date, so inside an ordinary expression (a `let`, a rate, a quantity's body) a
Calendar duration added to or subtracted from an instant is **E321**. Step a
date by calendar months or years with `add_calendar_months` or
`add_calendar_years` (§2.3), or offset it by a number of days.

**Instant positions.** A position that holds a date is an _instant position_:

- `simulate { from, to }` and a scenario's `simulate { to }`;
- an `at [...]` entry of an intervention or event, and a recurring schedule's
  `from` and `to`;
- an output's `trajectories { at }` and an observation's
  `emit_schedule = at [...]` entries;
- `value_at`'s time argument, a forcing's `breakpoints`, a bspline `window`, an
  interpolated forcing's `times`;
- the bounds of an `instant`-kind parameter, and a value given to one: a
  scenario's `set`, and a prior's `mu`, `mean`, `lower` or `upper` (a prior's
  scale, such as `sigma`, is a length);
- the operand compared with model time `t`, or with `t ± d`, in any expression.

In anchored mode a bare duration `D` in an instant position denotes
`origin + D`, and a comparison of the time since or until an instant `X` with a
duration (`t - X ⋈ D`, `X - t ⋈ D`, `-(X - t) ⋈ D`) compares `t` with the date
`X + D` (or `X - D`). Instant positions accept instant arithmetic
(`to = origin + 90 'days`, `at = [date("2020-03-01") + 7 'days]`), folded when
the model is compiled, and every `at` list accepts `date_range(...)`.

**The one-step calendar rule.** A calendar duration in an instant position,
written directly or through a `let`, is resolved as one calendar step: a whole
number of months (a year is twelve months) from a date known when the model is
compiled (`origin`, a `date(...)` literal, or a `let` of one) is the calendar
date `add_calendar_months` gives. From `origin = date("2020-01-01")`,
`simulate { to = 2 'years }` is 2022-01-01 (`t = 731`), `at [6 'months]` is
2020-07-01 (`t = 182`), and `1.5 'years` and `1 'years + 6 'months` are each one
step of 18 months. Any other calendar duration in an instant position is
**E321**, with a hint naming the explicit form:

- a fractional month (`0.5 'months`);
- a calendar step mixed with days, `date("2020-02-26") + 1 'years + 3 'days`,
  whose two orders of application give 2021-03-01 and 2021-02-28;
- two calendar steps (`origin + 6 'months + 6 'months`, or a step applied to an
  `add_calendar_*` result);
- a step from a date known only when the model runs: model time `t`
  (`t - 1 'months`), an `instant` parameter, or an observation anchor.

A calendar duration as a cadence (a recurring `every`, an output or emit
`every`) is **E322**; its hint names
`date_range(<start>, <end>, calendar_months = k)` in an `at` list. A _length_
keeps the mean month or year: `dt`, a forcing's `period`, `step` and `lag`, a
sinusoidal `phase`, a `via` mean, the bounds of a `duration` parameter, and a
unit literal in an ordinary expression.

**Instant arithmetic.** An instant moves by a duration, and the difference of two
instants is an exact duration. A **definite** instant is one built from
`date(...)`, `origin`, `add_calendar_*`, `instant`-kind parameters and `let`s of
these, moved by durations. In anchored mode six forms on a definite instant are
**E360**:

| Expression                                              | Result                   |
| ------------------------------------------------------- | ------------------------ |
| `date(b) - date(a)`                                     | Exact duration           |
| `date(a) + 30 'days`, `origin - d`                      | Instant (`d` a duration) |
| `date(a) + date(b)`                                     | **E360**                 |
| `sum(i in d, t0[i])`, with `t0` an instant              | **E360**                 |
| `2 * date(a)`, `date(a) / 2`, `date(a)^2`, `-date(a)`   | **E360**                 |
| `7 'days - date(a)`                                     | **E360**                 |
| `date(a) + 5`, `date(a) + 5 * 2`                        | **E360**                 |
| `if c then date(a) else 5 'days`                        | **E360**                 |

An offset is judged by its dimension, not its spelling: a constant with no unit
would be read in `time_unit`, so it is refused and written `5 'days`. A
conditional is written on the duration, `origin + (if c then 5 'days else 7
'days)`. An expression that is not a definite instant is not checked: model time
`t` is not one, so a rate term such as `k * (t0 - t)` is accepted, and so is a
function call other than `date(...)` and `add_calendar_*`, such as
`max(date(a), date(b))`.

In **unanchored mode** (no `origin`) there is no calendar, the refinement is
inactive, and a calendar duration is its mean length everywhere, including rate
expressions, table values and parameter bounds. Under `time_unit = 'months`, a
`let` of `6 'months` or `1 / (6 'months)` is six months, or its reciprocal, of
the mean length.

The treatment of dates across the language, data files and output is specified
in [`docs/dates.md`](dates.md).

### 2.2 Dimensional Type System

Every expression has a dimension, a product of powers of population P and time
T. Durations and rates are distinct dimensions, and the compiler refuses an
expression that combines them inconsistently:

```camdl
'days     — dimension: time
'per_day  — dimension: 1/time (rate)
```

```camdl
5 'days + 3 'days         → 8 'days (time + time = time)
1 / (14 'days)            → rate (1/time)
0.1 'per_day * 100        → 10 'per_day (rate × scalar = rate)
0.1 'per_day * 5 'days    → 0.5 (rate × time = dimensionless) ✓
```

```camdl
5 'days + 0.1 'per_day    → ERROR: cannot add time and rate
```

A sum whose operands have different dimensions is **E302**. A zero (`0`, `0.0`)
takes any dimension (§2.2.1).

### 2.2.1 Transition Rate Dimensional Analysis

Every transition rate must have dimension **P·T⁻¹**, a population per unit time
(§1). A per-capita rate written where a total propensity belongs has dimension
T⁻¹ and is refused:

```camdl
# ✓ Correct: beta:T⁻¹ * S:P * I:P / N:P = P·T⁻¹
infection : S --> I @ beta * S * I / N

# ✗ Error E300: beta:T⁻¹ * I:P / N:P = T⁻¹ (missing S)
infection : S --> I @ beta * I / N
#   error[E300]: transition 'infection' rate has wrong dimension
#     expected: P·T⁻¹ (population-level rate)
#     got:      T⁻¹ (per-capita rate)
```

The dimension of a leaf is fixed by what it is: a compartment reference is P;
model time `t` is T; a parameter of kind `rate` is T⁻¹, `probability` is 1,
`count` is P, `instant` and `duration` are T; a unit literal has its unit's
dimension (`'days` T, `'per_day` T⁻¹, `'count` P). Multiplication adds
exponents and division subtracts them.

**Inference.** Every expression-bearing position of the model contributes to one
system of equations over the exponents of P and T: a product's dimension is the
sum of its factors', the operands of `+`, `min` or a comparison have one
dimension, the argument of `exp`, `log` or a trigonometric function is
dimensionless, and each position contributes the dimension it requires. The
system is solved as a whole, following Kennedy ("Dimension types", ESOP 1994),
so a parameter whose dimension follows only from several uses jointly (`a * b`
in one rate, `a / b` in another) is resolved, and an operator has one meaning
wherever it appears. A `positive` or `real` parameter takes its dimension from
this system; when the system leaves it undetermined, the compiler reports
**I300**, and a `dim` clause states it (§4.1.1). A parameter used as a rate in
one place and as a count in another is a contradiction, reported where it
surfaces, with the conflicting use named. A system with a rational but no
integer solution (`(kappa * w)^2 * I` as a rate needs `kappa · w = T^-1/2`) is an
error.

**What each position requires.**

| Position                                                                              | Dimension                                     | Code |
| ------------------------------------------------------------------------------------- | --------------------------------------------- | ---- |
| transition rate                                                                       | P·T⁻¹                                         | E300 |
| overdispersion σ² in `overdispersed(rate, σ²)`                                        | T (the noise's Gamma shape is dt/σ²)          | E308 |
| ODE derivative                                                                        | P·T⁻¹                                         | E306 |
| balance expression                                                                    | P                                             | E305 |
| forcing `lag`; periods, phases, breakpoints, knots; smooth `per`                      | T                                             | E309, E312 |
| forcing values, amplitude, baseline, harmonics, identity-link spline coefficients     | the forcing's declared dimension D            | E312 |
| log-link spline coefficients                                                          | 1                                             | E301 |
| smooth level, slope, scale, modes                                                     | log link: 1, T⁻¹, 1, 1; identity link: D, D·T⁻¹, D, 1 | E301, E312 |
| initial value; initial Poisson `rate`, Binomial `n`, NegBinomial `mean`, Normal `mean` and `sd` | P                                  | E313 |
| initial Binomial `p`; NegBinomial `dispersion`                                        | 1                                             | E313 |
| transfer `fraction`                                                                   | 1                                             | E314 |
| transfer `count`; an assigned value; an added count                                   | P                                             | E314 |
| an intervention's time expression                                                     | T                                             | E315 |
| a time read as a number: schedule `every`/`from`/`to`/`at_day`, a trigger `window`, `after`, `cooldown`, `simulate`, a scenario's `to`, output and emit times | T | E315 |
| a reactive trigger's threshold                                                        | the observed stream's scored column           | E315 |
| Poisson `rate`; NegBinomial and zero-inflated `mean`; Binomial and BetaBinomial `n`   | P                                             | E304 |
| Binomial and Bernoulli `p`; Beta `mean`; zero-inflation `pi`; shapes and concentrations | 1                                           | E304 |
| NegBinomial and zero-inflated `dispersion`                                            | 1                                             | E307 |
| Normal `mean`                                                                         | the scored column's dimension                 | E304 |
| Normal `sd`                                                                           | the mean's; on a `count` column only its T exponent (below) | E304 |
| a quantity's threshold (`count_above`, `first_above`, …)                              | the series' dimension                         | E317 |
| a `value_at` time                                                                     | T                                             | E317 |
| the arms of a contrast difference                                                     | one dimension                                 | E297 |
| prior arguments                                                                       | unit literals only (§2.2.3)                   | E316 |
| parameter bounds `in [lo, hi]`                                                        | unit literals only, in θ's dimension          | E316 |
| a table value, in a table with a unit annotation (`mu : age 'per_day = [...]`)        | the annotation's dimension                    | E302 |

Beyond the positions: a sum whose operands differ is **E302**; the argument of
`exp()` or `log()`, a power's exponent, and a base under a non-constant exponent
must be dimensionless (**E301**); and a square root, or a power `x^(p/q)`,
needs dimension exponents that 2, or q, divides (**E304**).

**Literals.** A numeric literal is dimensionless, with two exceptions. A zero
takes any dimension, and so does a product with a zero factor or a quotient with
a zero numerator (`0 * I`, `0 / I`). And a literal, or a constant expression of
literals (arithmetic, `min`/`max`, `sqrt` and the other functions of numbers),
that is a whole additive term of a rate, derivative, balance, initial value,
count-valued likelihood argument, intervention count or quantity threshold takes
that position's dimension: `I = 10`, `S = N0 - 50`, `S = 0.95 * 10000`,
`rate = 100`, and a constant importation `beta * S * I / N + 0.5`. Inside a
product, a quotient, or a function of anything that is not constant, a literal
is a pure number: `(1/730) * R` is not a rate (**E300**), and in
`(1000 - S) / 1000` the 1000 is not a count (**E302**).

A literal takes its dimension from its position and never from the expression it
is combined with. Inferring it from a neighbour would accept
`(1000 - S) / 1000`, but would equally accept `I + 1e-6` in a rate, a
dimensionless constant added to a count by mistake. A number that is a quantity
is therefore written as one, in one of two forms:

- a **typed constant**, for a named quantity used more than once:
  `let N0 : count = 1000`, then `final((N0 - S) / N0)`;
  `let waning : rate = 1 / (730 'days)`, then `waning * R`;
- a **unit literal**, for a single value: `final(1 - S / (1000 'count))`,
  `R / (730 'days)`.

A coefficient whose dimension has no unit literal is a typed constant with a
`dim` clause: `let b : positive dim P^-1*T^-1 = 0.0001`. The E302 hint for a bare
number names both forms with the kind and unit that fit.

**Powers.** In `a^n` with `n` an integer known when the model is compiled (a
literal, a typed constant, or arithmetic of them), the result has `n` times
`a`'s dimension. A compile-time fraction `p/q` requires
`q · dim(result) = p · dim(a)` to have an integer solution: `(I / N)^0.5` is
accepted, and `I^0.5` in a rate is not. Under any other exponent the base must
be dimensionless, since `d^y` with `y` varying during a fit has no dimension;
`unchecked_dim` (§2.2.2) asserts a dimension for a deliberate exception.

**Count noise.** The `sd` of a `normal` likelihood on a `count` column must have
the mean's T exponent, and its P exponent is free. A variance model for counts
adds a term in the count to a term in its square (He et al. 2010,
ρ(1−ρ)C + ρ²ψ²C²); the sum is consistent because a count is a number of
individuals, not a quantity whose unit can be rescaled. The exemption covers the
`sd`'s sums, products, powers and roots. The argument of `exp`, `log` or a
trigonometric function stays dimensionless, so `sd = exp(projected)` is
**E301**. On a `real` column the `sd` has exactly the mean's dimension, and the
column's dimension is inferred from the two.

**Scope of the check.** Every expression that reaches the dynamics,
observations, interventions, initial conditions and quantities is checked
against its position. The check does not cover the scale of a parameter value
supplied at run time (always in `time_unit`), the inside of an `unchecked_dim`,
or the physical meaning of a prior (§2.2.3).

`camdlc check --no-dim-check`, and the same flag on a compile, disables the
analysis; it is for a confirmed false positive, which is a compiler defect to
report.

### 2.2.2 Asserted dimensions: `unchecked_dim`

`unchecked_dim` asserts the dimension of an expression that dimensional analysis
cannot type. The case it exists for is the α-mixing term `(I + ι)^α` of He et
al. (2010), with a non-integer `α`: `P^0.976` is not a dimension, and the
formulation is established in the literature.

```
unchecked_dim(EXPR, dim = NAME, reason = "…")
```

```camdl-doctest-preamble unchecked-dim
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
```

```camdl preamble=unchecked-dim
transitions {
  infection : S --> E
    @ beta(t) * unchecked_dim((I + iota)^alpha,
                              dim = population,
                              reason = "He et al. 2010 α-mixing exponent")
             * S / pop(t)
}
```

**Meaning.** The call has dimension `NAME`, and the expressions around it are
checked as usual. The wrapped expression is not checked. When the model runs,
the call is the value of `EXPR`.

`dim` is one of `dimensionless`, `population`, `time`, `rate`,
`population_rate` and `per_population`. The asserted dimension is the one that
makes the surrounding expression correct. Above, `beta(t)` is T⁻¹ and
`S / pop(t)` is dimensionless, so the call must be P for the rate to be P·T⁻¹:
`dim = population`. Asserting `dimensionless` there leaves the rate at T⁻¹ and
is **E300**.

**Refused.** A call without a `reason` string, or with a `dim` name not in the
list, is **E240**.

References:

- He, Ionides, & King (2010). _J. R. Soc. Interface_ 7(43): 271–283.
  doi:10.1098/rsif.2009.0151.
- Bretó, He, Ionides, & King (2009). _Annals of Applied Statistics_ 3(1):
  319–348. doi:10.1214/08-AOAS201.

### 2.2.3 Dimensions and priors

A prior is a probability distribution over the number a parameter takes in the
model's units. Dimensional analysis checks priors only for unit conversions; it
never derives a parameter's dimension from a prior.

**Values in a prior.** A parameter's dimension comes from its declaration and its
uses in the model. Choosing a prior family neither establishes nor restricts it:
`gamma : rate ~ beta(alpha = 2, beta = 2)` is a valid prior on a recovery rate
whose value lies in (0, 1) per `time_unit`. A bare number in a prior is a number
in model units: `normal(mu = 0.2, sigma = 0.1)` on a rate is 0.2 per day under
`'days` and 0.2 per week under `'weeks`. A parameter inside a prior argument is
also a number, so arithmetic on hyperparameters is unrestricted:
`log_normal(mu = log(m_h), ...)` with `m_h : rate` takes the logarithm of
`m_h`'s value.

**Unit literals in a prior.** A unit literal requests a conversion, which the
compiler applies: under `'weeks`, `normal(mu = 0.2 'per_day)` has mean 1.4, and
`log_normal(mu = log(0.2 'per_day))` converts the rate before taking its
logarithm. A conversion is correct only in the argument's own dimension, so a
unit literal whose dimension disagrees with the argument is **E316**. Each whole
term of an argument is read on its own: in `mu = m_h + 1 'days` the literal
asserts a duration, whatever `m_h` is. The arguments' dimensions are:

| Family                        | Argument                    | Dimension |
| ----------------------------- | --------------------------- | --------- |
| normal, uniform, half-normal, log-uniform, truncated normal | location, scale, bounds | θ's |
| exponential; gamma            | `rate`                      | θ's inverse |
| gamma; beta                   | `shape`; `alpha`, `beta`    | 1 |
| log-normal                    | `mu`, `sigma`               | 1 (they are the mean and sd of ln θ) |

`log_normal(mu = -2 'per_day, ...)` passes a rate as a log-coordinate and is
E316; so is `normal(mu = 0.2 'days)` on a rate, and so are two unit literals
that disagree within one argument. When the model does not determine θ's
dimension, a literal is converted by its own unit and accepted.

**I316.** These dimensions also drive one lint, **I316** (Info): a log-normal
`mu` or `sigma` that is directly a parameter with a dimension, which is probably
a rate meant as `log(m)` or a log-coordinate that should be declared
`real dim 1`. The prior arguments' dimensions never constrain a parameter's
dimension, and a parameter used only in priors needs none (it raises no I300).

**Changing `time_unit`.** A prior written with unit literals in the positions of
the table describes the same belief under any `time_unit`; a prior written with
bare numbers does not, and must be rewritten when the unit changes. For a
log-normal, μ shifts by the logarithm of the conversion factor (+ln 7 from days
to weeks) and σ is unchanged.

A prior is a measure on θ; its density has units of 1/[θ]. Under a change of
unit the measure is carried through by the Jacobian of the change of
coordinates, and the density's functional form need not survive, so a family's
form says nothing about θ's dimension. The Jeffreys prior for a scale
parameter, p(θ) ∝ 1/θ, is the form that is invariant under rescaling (Jeffreys,
H. 1946, "An invariant form for the prior probability in estimation problems",
_Proc. R. Soc. Lond. A_ 186:453–461); camdl does not require it.

### 2.3 Date Literals

`date("YYYY-MM-DD")` is an ISO 8601 date, read as its offset from the model's
`origin` in `time_unit`. The calendar is proleptic Gregorian. Dates across the
language, data files and output are specified in [`docs/dates.md`](dates.md).

```camdl preamble=sir-basic
origin = date("2019-01-01")   # top-level declaration (optional)

simulate {
  to = date("2021-06-30")     # 911 days from origin (in time_unit = 'days)
}
```

**Meaning.** Under `time_unit = 'days` the offset is the whole number of days
from `origin`; under another unit it is that number divided by the unit's length
in days (§2.1). `origin` sets the coordinate between dates and model time and has
no effect on the dynamics. The IR stores it as the ISO string (`"origin"`) and
as an integer day number (`"origin_rata_die"`).

Dates also appear outside `date()` literals: an observation data file may have
an ISO-date time column, converted through `origin` and `time_unit`;
`camdl simulate --dates` adds a calendar date column to its output; `camdl fit
summary` prints `instant`-kind estimates as dates; and the `instant` and
`duration` parameter kinds have dimension T (§4.1).

**Refused.** `date(...)` in a model with no `origin = date(...)` is **E220**:

```camdl
# ERROR E220: date("2021-06-30") used but no 'origin' declared
simulate { to = date("2021-06-30") }
```

#### Calendar arithmetic and `origin`

In anchored mode two functions step a date by calendar months or years. They
are evaluated when the model is compiled and are accepted only where a constant
instant is:

```camdl
add_calendar_months(d, n)   # Instant × Int → Instant
add_calendar_years(d, n)    # Instant × Int → Instant
```

`d` is a constant instant: a `date(...)` literal, `origin`, or a nested
`add_calendar_*` call. `n` is a constant integer. The step is proleptic
Gregorian `(year, month, day)` arithmetic with **month-end clamping**:
`add_calendar_months(date("2020-01-31"), 1)` is `date("2020-02-29")` (a leap
year), and `add_calendar_months(date("2021-01-31"), 1)` is `date("2021-02-28")`.
Neither function uses the mean month length.

In anchored mode `origin` is a reserved name for the origin as a constant
instant, accepted wherever a constant instant is: `simulate { from = origin }`,
`at [origin, ...]`, `add_calendar_months(origin, 6)`, `origin + 90 'days`,
`let landmark = origin + 90 'days`, and the operand compared with `t`
(`t > origin + 30 'days`). It is not model state and is not a parameter.

`date_range(...)` generates a list of instants from a start, an end or a count,
and a cadence, for use in an `at` list; its forms are specified in
[`docs/dates.md`](dates.md).

**Refused.** In unanchored mode, a reference to `origin` and a call to
`add_calendar_months` or `add_calendar_years` are **E327**.

### 2.4 Three tiers of dimensional information

A declaration states dimensional information at one of three levels of detail:

| Tier                  | Syntax                                                                    | Carries                        | Used for                                                                                                          |
| --------------------- | ------------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| 1. **Kind keyword**   | `rate`, `probability`, `count`, `positive`, `real`, `instant`, `duration` | a dimension, from the kind     | parameters; `instant` and `duration` are time-typed (T), see [`dates.md`](dates.md)                               |
| 2. **`dim` clause**   | `dim T`, `dim T^-1`, `dim P`, `dim P/T`, `dim 1`                          | a dimension only               | a `real` or `positive` parameter whose uses do not determine its dimension (§4.1.1)                               |
| 3. **Unit literal**   | `'days`, `'years`, `'per_day`, `'per_year`, `'count`, `'ratio`            | a dimension and a scale        | concrete values: `time_unit`, duration literals, table entries, forcing declarations; never a parameter           |

A `dim` clause states a dimension for a value supplied later (from `--params`,
a prior or a fit), which is always in the model's `time_unit`, so it carries no
scale: `dim T` is a time, not days or years. A unit literal states the
dimension and the scale of a value written in the model, and the compiler
converts it to `time_unit`. The two cannot be merged into one syntax: a
parameter's value has no scale of its own for a unit literal to convert, so
`beta : positive 'per_day` is **E117**, with the fix `beta : positive dim 1/T`
(§4.1.1).

Every unit literal decomposes into a dimension and a scale. Its dimension plays
the role of a `dim` clause, and its scale drives the conversion, so a unit
literal carries strictly more than the `dim` clause of the same dimension:

```
'years       ⊂    dim T   ⊂   (any time-dimensioned value)
(dim+scale)       (dim)         (unconstrained)
```

A kind keyword names a common dimension (`rate` is `dim T^-1`, `probability` is
`dim 1` with domain [0, 1], `count` is `dim P`), so most parameters need no `dim`
clause. §2.5 and §6.1 show unit literals on table values, §4.1.1 the `dim`
clause on parameters, and §7 the unit literal every forcing declaration
requires.

**`'ratio` and `probability`.** Both are dimensionless. `'ratio` is a tier-3
annotation on a value with no bound, such as a seasonal multiplier of 0.7, 1.3 or
50. `probability` is a parameter kind with domain [0, 1] and a logit transform
under inference (§4.1).

### 2.5 Table Unit Annotations

A table carries one unit for all its values:

```camdl
tables {
  fertility  : age 'per_day   = [0.0, 0.02]
  age_dur    : age 'years     = [5, 60]
  C_age      : age × age      = [[12.0, 4.0], [4.0, 8.0]]  # dimensionless
}
```

`fertility : age 'per_day` declares every value of the table in `'per_day`; the
compiler converts the values to `time_unit` (under `'days`, `age_dur` is stored
as 1826.2125 and 21914.55). A dimensionless table, such as a contact matrix or
a set of weights, has no annotation. A table value whose dimension disagrees
with the annotation is **E302** (§2.2.1). Tables are specified in §6.

---

## 3. Compartments

A compartment is a population state of the model.

```camdl
compartments { S, E, I, R }
```

A compartment is an integer-valued count unless declared `: real`, which makes
it a continuous-valued state whose dynamics are written in the `ode { }` block
(§11):

```camdl-doctest-preamble comp-real
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
```

```camdl preamble=comp-real
compartments {
  S, I, R,
  W : real       # continuous-valued (environmental reservoir)
}
```

A stratified compartment has one cell per combination of its dimensions' levels
(§5). A cell is read by indexing, `S[child]` or `S[child, female]`; a bare `S` in
model dynamics is the total over all cells (§1).

Related sections: §5 (stratification), §9 (transitions), §11 (ODE block), §15
(initial conditions).

---

## 4. Parameters

A parameter is a named model input whose value is supplied when the model is
run or fitted.

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

A declaration is, in order: the name (with its index dimensions, §4.3), the kind
(§4.1), an optional `dim` clause (§4.1.1), optional bounds (§4.4) and an
optional prior (§20.2). It never holds a value; values are supplied as described
in §4.2.

### 4.1 Parameter Types

```
rate        : ≥ 0, dimension 1/time. Default transform, by its bounds: θ = eᶻ
              on [0, ∞); θ = lo + eᶻ on [lo, ∞); log-interval, ln θ = ln lo +
              ln(hi/lo)·σ(z), on [lo, hi] with lo > 0; interval, θ = lo +
              (hi − lo)·σ(z), on [0, hi].
probability : ∈ [0, 1], dimensionless. Default transform: logit. The [0, 1] is
              carried by the type: a `probability` that declares no `in [lo, hi]`
              is searched over [0, 1] by inference, and a supplied value outside
              [0, 1] is rejected. It is the only kind whose type pins a finite
              interval — see §4.4.
positive    : > 0 under inference (θ = eᶻ). The kind fixes no dimension: it is inferred from the
              parameter's uses unless a `dim` clause states it (§4.1.1).
              Default transform: as `rate` — θ = eᶻ on [0, ∞), θ = lo + eᶻ on
              [lo, ∞), log-interval on [lo, hi] with lo > 0, interval on
              [0, hi].
count       : dimension P (a population count). Integrality is not enforced —
              `let iota : count = 1e-6` compiles; the type carries the [P]
              dimension, not an integer constraint.
real        : unconstrained (default if omitted).
instant     : dimension [T], absolute time. Renders as a date in anchored
              mode (requires origin). Negative lower bounds allowed.
duration    : dimension [T], a span. Renders as a span.
```

A kind fixes the parameter's dimension, its search range under inference when it
declares no bounds, and the transform inference uses (§4.4). `count` and
`duration` take the same transforms as `rate`, and `real` and `instant` the
identity, or a one- or two-sided map when bounded. σ is the logistic function.

The search range of `rate`, `positive`, `count` and `duration` is [0, ∞). A
value supplied for a parameter of these kinds is checked against declared
bounds only; with no `in [lo, hi]`, a negative value is not refused when
parameter values are resolved, and a negative rate fails only when it makes a
transition's propensity negative.

An initial state is given in the `init { }` block (§15), and is estimated with
the fit setting `ic_free`; there is no parameter kind for it.

**`instant` and `duration`.** Both have dimension T. In anchored mode an
`instant` is an instant (§2.1): it moves by an exact duration, the difference of
two instants is an exact duration, and adding two instants, scaling one, or
subtracting one from a duration is **E360**. A reference to a `duration`
parameter is always Exact, even when its bounds are written in `'months`
(`delay : duration in [1 'months, 6 'months]`): the bounds are lengths, and do
not make the parameter's uses Calendar. In unanchored mode both kinds are plain
T-dimensioned values.

### 4.1.1 The `dim` clause

A `dim` clause states the dimension of a `positive` or `real` parameter, whose
kind does not fix one. Usually the parameter's uses determine it: in
`beta * S * I / N` the rate is P·T⁻¹, so a `positive` `beta` is T⁻¹. When the uses
leave it undetermined (**I300**), or to have the dimension stated and checked,
write the clause after the kind:

```camdl
parameters {
  beta      : rate                      # dimension T⁻¹, fixed by the kind
  amplitude : real dim 1                # dimensionless
  mu        : positive dim 1/T          # per-capita rate
  coupling  : positive dim P/T          # population-level rate
  b         : real dim T^-1 in [-1, 1]  # a rate-scale spline coefficient, may be negative
  lag       : positive dim T            # a duration
}
```

A dimension is a product of powers of `P` (population) and `T` (time), written
with `*`, `/` and integer exponents:

| Clause                  | Dimension     | Domain name            |
| ----------------------- | ------------- | ---------------------- |
| `dim 1`                 | dimensionless | probability, ratio, R₀ |
| `dim P`                 | population    | count                  |
| `dim T`                 | time          | duration               |
| `dim 1/T`, `dim T^-1`   | T⁻¹           | per-capita rate        |
| `dim P/T`, `dim P*T^-1` | P·T⁻¹         | population-level rate  |
| `dim P^2/T`, …          | P²·T⁻¹, …     | (any other product)    |

**Meaning.** The clause fixes the parameter's dimension, so a use that
contradicts it is an error: `tau : positive dim 1` added to a population
(`I + tau`) is **E302**. The clause carries no scale; the parameter's value is
in `time_unit` (§2.4).

**Refused.**

- A `dim` clause on any kind other than `positive` and `real` is **E281**,
  whether it repeats the kind's dimension (`rate dim T^-1`) or contradicts it
  (`rate dim 1`).
- Brackets after a kind give a parameter family's size, never a dimension. A
  dimension in brackets (`real [1/T]`, `positive [P*T^-1]`) is
  **E116**, with the fix `real dim 1/T`. A size below two (`real[1]`) is
  **E417**: one cell is a scalar, written `real`, and `[1]` meant as a
  dimension is `real dim 1`.
- A unit literal on a parameter (`gamma : positive 'per_year`) is **E117**, with
  the fix `positive dim 1/T`, because a parameter's value has no scale for the
  literal to convert.

Under `--json-errors` each of these diagnostics carries a machine-applicable
fix-it (`"fixes": [{"loc": …, "replacement": …}]`).

### 4.2 External parameter values

A parameter's value is supplied when the model is run, from outside the model
file, or from a scenario's `set = { … }` inside it (§17). The order in which the
sources override each other is specified in the run spec (§1.3).

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

The TOML format is specified in §20. A scalar computed outside the model (a
demographic rate from a pipeline, a constant from the literature) belongs in a
generated `--params` file, which the model then reads directly. A table (§6)
holds indexed covariate data and has at least one dimension; it does not hold a
scalar.

### 4.3 Indexed Parameters

A parameter declared with one or more dimensions is a family of scalar
parameters, one per level, or per cell of the cartesian product of several
dimensions:

```camdl
parameters {
  gamma               : rate
  N[patch]            : positive   # expands to N_urban, N_rural, ...
  R0[patch]           : positive   # expands to R0_urban, R0_rural, ...
  mu[village, season] : rate       # expands to mu_kwaru_wet, mu_kwaru_dry, ...
}
```

**Meaning.** Over `village = [kwaru, ajura]` and `season = [wet, dry]`,
`mu[village, season]` is the four scalar parameters `mu_kwaru_wet`,
`mu_kwaru_dry`, `mu_ajura_wet` and `mu_ajura_dry`, each with the declared kind,
bounds and prior. A cell's name is `<base>_<level1>_<level2>_…`, in the order
the dimensions are written. Each index names a dimension declared in
`dimensions { }` (§5); the parameter's dimensions need not be stratification
dimensions of any compartment. In the IR the family is its scalar parameters.

**Indexing.** A reference gives every dimension, in declaration order:

```camdl
let C[v in village, s in season] = mu[v, s] * gamma   # mu[v,s] → Param("mu_kwaru_wet") etc.
```

```camdl
let beta[p in patch] = R0[p] * gamma   # R0[p] → Param("R0_urban") etc.
```

Inside `[...]` on a parameter reference a name is resolved as a bound index
variable (such as `p`) or as a level of the dimension (`R0[urban]` is
`R0_urban`), and never as a `let` binding or another parameter: `R0[urban]` is
the level `urban` even when a `let` named `urban` exists. The indexed form is
accepted wherever a parameter name is: in expressions (§9.6), `init` blocks
(§15.2), and a scenario's `set` and `scale` (§17.1). At run time,
`--param-vec PREFIX=FILE` supplies a family's values from a two-column
`name<TAB>value` file, each row naming `PREFIX_<key>` (§21):

```bash
camdl simulate model.ir.json --param-vec R0=/tmp/r0_posterior.tsv
```

**Refused.** An unknown index dimension is **E330**, and a repeated one
(`mu[village, village]`) is **E331**. A `let` binding whose name is a level of
any dimension raises the warning **W103**:

```camdl
let urban = 1.0   # W103: let binding 'urban' shadows stratum value 'urban'
                  #   in dimension 'patch'. This is allowed but consider renaming.
```

### 4.4 Parameter Bounds

An `in [lo, hi]` clause states the range of a parameter's values:

```camdl
parameters {
  R0       : positive in [1.0, 20.0]    # scalar with bounds
  rho      : probability in [0.0, 1.0]  # redundant but explicit
  R0[patch]: positive in [0.5, 15.0]    # all strata get same bounds
  gamma    : rate                        # unbounded (beyond type constraint)
}
```

**Meaning.** The bounds apply to every scalar of an indexed family and are
stored in the IR. Inference searches within them and chooses its transform from
them (§4.1). A supplied value outside them is refused when parameter values are
resolved; the compiler never sees values, so the check is made by the runtime,
and a simulation does not clamp a value during a run.

A parameter with no `in [lo, hi]` takes the range its kind carries.
`probability` carries [0, 1], so the `rho` line above declares what its kind
already does, with the same search range and the same logit transform. No other
kind carries a finite range (`rate`, `positive`, `count` and `duration` are
[0, ∞), `real` and `instant` unbounded), so for those kinds `in [lo, hi]` is
the only way to give inference a finite search range. The bounds of an
`instant` parameter are an instant position (§2.1).

Related sections: §17 (scenarios), §20 (parameter files and priors), §21.5
(inference).

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

```camdl-doctest-preamble partial-strat
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
```

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
| indexable?                    | yes — `I[child]`            | by location only — `stage(I[child], 2)` (§9.4.1)       |
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
- **Where a value is scored against data, a bare name may pool residence
  structure but not a population stratum** (`E280`, §12.1). In model dynamics
  a bare `I` is the total across every axis, and that is unchanged.

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

```camdl-doctest-preamble tables-demo
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
```

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

A table read with `read("path")` is loaded at compile time and inlined into the
IR, so the IR names no file and the runtime opens none. The canonical IR format
is compact one-element-per-line JSON (a `--pretty` variant exists for
inspection); there is no binary format.

A table declared `external("NAME")` is the exception: its values are chosen when
the model is run, not when it compiles, so one model can be run or fitted under
several versions of a data input.

```camdl
tables {
  C : patch × patch = external("contact")
}
```

The file is bound with `--table contact=FILE` on the command line, or with
`[data.tables] contact = "FILE"` in a fit config (see
[`docs/fit-toml.md`](fit-toml.md)); the command line wins where both bind the
same name. The file holds the table's cells as numbers in row-major order (the
last index varies fastest): a CSV or TSV of numbers with no header, blank and
`#` lines skipped, or a JSON array. It must hold exactly one number per cell —
the product of the declared dimensions' sizes — and a file with more or fewer is
refused, naming the table and both counts. The model's identity covers the
declaration (`external("contact")` and its dimensions); the file's bytes are an
input of each run, so the same model run under two files is one model and two
runs. A run with an external table left unbound, or with a binding for a name
the model does not declare, is refused.

### 6.6 Parameterized Table Entries

Inline table values can be parameter names or arithmetic expressions, not just
numeric literals:

```camdl-doctest-preamble table-bsex
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
```

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

Named time-dependent forcing functions, usable in rate expressions. Seven
built-in types cover real-world needs:

- `sinusoidal` — smooth seasonal forcing
- `periodic` — repeating step function (day-of-week, month-of-year effects)
- `piecewise` — non-repeating step function (policy changes, campaign windows)
- `interpolated` — data-driven time series (empirical covariates)
- `fourier` — finite Fourier series with estimable cos/sin harmonic pairs
  (`period`, `harmonics = [[a1, b1], [a2, b2], ...]` — each harmonic is a
  2-element list), for smooth periodic forcing richer than a single sinusoid
  (gh#59)
- `periodic_spline` — periodic B-spline with uniform knots (`period`,
  `n_basis`, `coefs` — a list of exactly `n_basis` coefficients — plus optional
  `degree`, default 3, at most 5 and less than `n_basis`, and optional
  `link = identity | log`), for flexible smooth seasonality (gh#59)
- `bspline` — a clamped B-spline over a window, an error to evaluate outside it
  unless overridden, for a smooth rate that changes through time without
  repeating; see
  [B-spline forcing](#b-spline-forcing-bspline) below

```camdl-doctest-preamble forcing-demo
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
```

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

  # Smooth periodic shapes richer than one sinusoid. `harmonics` is a list of
  # [cos, sin] coefficient pairs, one per harmonic of the base period.
  seasonal_fourier : fourier 'ratio {
    period    = 365.25 'days
    harmonics = [[0.30, 0.10], [0.05, 0.02]]
  }

  # `coefs` is required and must have exactly n_basis entries; `degree`
  # defaults to 3 and must be less than n_basis.
  seasonal_spline : periodic_spline 'ratio {
    period  = 365.25 'days
    n_basis = 6
    degree  = 3
    coefs   = [1.0, 1.2, 1.4, 1.1, 0.9, 0.8]
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
  windows). Range endpoints are model time in the model's `time_unit`, whole
  units and inclusive, whether written with a unit (`7 'days : 13 'days`) or
  bare (`7:13`); they are not bin indices. The compiler converts them to bins
  of width `step`: bin `i` covers `i·step` to `(i+1)·step − 1`. With 1-day
  steps an endpoint is its own bin. With wider bins each range must start on a
  bin edge and end on the last unit of a bin: with `period = 364 'days` and
  `step = 7 'days`, `7:13` switches on bin 1 and `0:13` bins 0 and 1, while
  `8:13` or `7:14` is **E406**, whose hint names the bin edges (ranges start
  at 0, 7, 14, … and end at 6, 13, 20, …). Each range is written low to high
  and lies inside the bins (`0:364` for a 365-day period with 1-day steps,
  `0:363` for a 364-day period with 7-day steps); a reversed range (`100:7`)
  or one that runs past the end of the last bin (`400:500`, or `300:500`,
  which is not read modulo the period) is **E406**. A window that spans the
  end of the period is written as two ranges (`100:364, 0:7`).
  **In anchored mode** (top-level `origin` declared), bare-numeric
  entries inside `on=[...]` are **E323** — anchored periodic schedules must use
  `date(...)` entries (or, for the rare legitimate "day-offset from origin"
  case, the `--time-format internal-days` opt-in on the data boundary). See
  [`docs/dates.md`](dates.md) and the typed-time proposal §3 (Rule 2 /
  bare-numeric subsection).

**Bin width at run time.** In both forms the runtime divides the period into
equal bins, each `period / n` wide, where `n` is the number of values: `t` falls
in bin `floor((t mod period) / (period / n))`. In the range form `n` is
`period / step` rounded to the nearest whole number, so when the period is not a
whole number of steps the bins are not `step` wide. `period = 365.25 'days` with
`step = 1 'days` makes 365 bins of 1.000685 days, not 1 day: bin `i` starts at
`i × 1.000685`, so its switch points fall partway through a day and drift from
midnight across the year (bin 182 starts at 182.125, three hours into day 182),
and each year restarts 0.25 days later on the day grid. A range is still
converted to bins as described above, by whole steps; only where those bins
start and end at run time moves. `camdl check` notes this as **I401**, with the
actual bin width. To keep the switch points on whole days, use a period that is
a whole number of steps (`period = 365 'days`); to keep the year at 365.25 days,
accept the drift, which is at most `step / 2` within a period.

Forcing functions are used in rate expressions by name or with explicit `(t)`:

```camdl-doctest-preamble forcing-school
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
```

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

### B-spline forcing (`bspline`)

A rate that changes smoothly through an outbreak — with behaviour, control
measures and the depletion of high-contact groups — but not on a known date and
not periodically, is a **B-spline**: a weighted sum f(t) = Σₖ cₖ Bₖ(t) of K
fixed, smooth, local bump functions (the basis), whose weights `coefs` are the
curve's coefficients. Each basis function is a piecewise polynomial of `degree`
p (default 3, cubic; at most 5) that is nonzero over at most p + 1 adjacent
knot intervals, so a coefficient moves the curve only near its own knots.

```camdl-doctest-preamble forcing-bspline
compartments { S, I, R }
parameters {
  gamma : rate
}
let N = S + I + R
transitions {
  infection : S --> I @ beta_t * S * I / N
  recovery  : I --> R @ gamma * I
}
```

```camdl preamble=forcing-bspline
forcing {
  # Equal knot spacing: n_basis - degree intervals over the window.
  beta_t : bspline 'per_day {
    window  = [0 'days, 120 'days]
    n_basis = 6
    coefs   = [0.30, 0.40, 0.55, 0.35, 0.20, 0.25]
  }
}
```

The knots take one of two forms, and exactly one must be given:

- **`window = [a, b]` with `n_basis`** — `n_basis − degree` equal intervals
  between a and b, and `n_basis` coefficients.
- **`breakpoints = [a, …, b]`** — your own strictly increasing knots, the first
  and last being the window; `n_basis` is then implied (`len(breakpoints) − 1 +
  degree`) and is refused if also written.

The basis is **clamped**: the window ends are repeated knots, so the curve
starts exactly at its first coefficient at a and ends exactly at its last at b.
**The forcing is defined on its window.** A run that evaluates it outside the
window is an error, naming the forcing, the window and the side it was
evaluated on, unless the run is given `--allow-forcing-extrapolation` (in a fit config,
`[config] allow_forcing_extrapolation = true`). Under that override the forcing
holds: for t before a it is its value at a, after b its value at b, so a run
past the window carries the curve's last value rather than an extrapolated
trend, and the run says so once per forcing with the warning
`forcing-outside-range`:

```
warning: forcing `beta_t` is defined on t = 0 to t = 120 (its window); the run evaluated it after the end of its window, and its value at t = 120 is held from t = 120 to the end of the run
```

With `lag = ℓ` the forcing is evaluated at t − ℓ, so the hold starts at b + ℓ;
the times in the message are those of the argument t − ℓ. The check reads the
times the run evaluated, so a forcing read only under a condition that is false
past the window is never refused.

**Window ends and breakpoints are times**, read like `piecewise` breakpoints: a
duration literal, a `date(...)` under an `origin`, or an observation anchor.
Ending the window before the reporting delay, so the held value is a coefficient
the data inform rather than the prior's extrapolation, is written

```camdl
forcing {
  beta_t : bspline 'per_day {
    window  = [0 'days, last_obs - 2 'weeks]
    n_basis = 8
    coefs   = [0.3, 0.4, 0.5, 0.45, 0.35, 0.3, 0.25, 0.25]
  }
}
```

and an anchored window that resolves out of order under some data vintage is
refused when the run resolves it.

**`link = log`** makes the forcing f(t) = s · exp(Σₖ cₖ Bₖ(t)), positive for
any coefficients. The coefficients are then dimensionless log-values (a
dimensioned one is **E301**), and the declared unit supplies the scale s:
`bspline 'per_week { link = log … }` under `time_unit = 'days` is exp(g)/7 per
day. `periodic_spline` accepts the same `link`. Because the basis sums to one at
every t, adding a constant to every coefficient multiplies a log-link forcing by
its exponential — a separately estimated intercept multiplying it would
therefore not be identified, and the coefficients should carry the level
themselves.

**Coefficients may be parameters**, or arithmetic over them (`coefs = [0.0,
c2, c3, -(c2 + c3)]`), in a `bspline` and in a `periodic_spline` alike. They
are evaluated live at every step, and the compiler emits their gradient: ∂f/∂cₖ
is basis function k at the forcing's evaluation time (times f itself under
`link = log`), so PGAS with NUTS, IF2 and the particle filter can all estimate
them. Only the knots — a window end, a breakpoint, a period — must be constant
(**E600**), because they place the basis functions themselves.

**`coefs = smooth(level, slope, scale, modes, per)`** writes the coefficients
of a `bspline` as a straight line plus penalized curvature, so the prior on the
curve says how much it may bend rather than where each coefficient sits.
**It is not yet available in this release:** `camdlc` compiles it, but every
`camdl` command refuses a model that uses it while its sampling is still being
measured. Until then write explicit coefficients and keep `n_basis` small (5 or
6), since nothing else ties neighbouring coefficients together.

```camdl-doctest-preamble forcing-bspline-smooth
dimensions { mode = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10] }
compartments { S, I, R }
parameters {
  gamma   : rate
  a_b     : real dim 1 ~ normal(mu = -1.9, sigma = 0.6)
  b_b     : real dim 1/T ~ normal(mu = 0.0, sigma = 0.005)
  tau_b   : positive dim 1 ~ exponential(rate = 5.99)
  z[mode] : real dim 1 ~ normal(mu = 0.0, sigma = 1.0)
}
let N = S + I + R
transitions {
  infection : S --> I @ beta_t * S * I / N
  recovery  : I --> R @ gamma * I
}
```

```camdl preamble=forcing-bspline-smooth
forcing {
  # a_b: log beta at mid-window; b_b: its slope per day; tau_b scales the
  # ten curvature modes z (n_basis - 2 of them).
  beta_t : bspline 'per_day {
    window  = [0 'days, 168 'days]
    n_basis = 12
    link    = log
    coefs   = smooth(level = a_b, slope = b_b, scale = tau_b, modes = z, per = 1 'weeks)
  }
}
```

The coefficients are w = a·1 + b(ξ − t_m) + (τ/κ)·M·z, with ξ the Greville
abscissae (the mean of each basis function's interior knots), t_m the
mid-window time, and M = Q₊L⁻ᵀ the whitening of the integrated-curvature
penalty Ω (Ωᵢⱼ = ∫ Bᵢ″ Bⱼ″ dt over the window): Q₊ spans the complement of
the lines {1, ξ} and L is the Cholesky factor of Q₊ᵀΩQ₊. The runtime builds M
from the resolved knots, so an anchored window works. Inside the window the
linear part of the curve is exactly a + b(t − t_m), and the curvature of the rest is
T_ref³ ∫ g″² dt = τ²‖z‖²/(K − 2), with κ = √(K − 2)·T_ref^{3/2} and T_ref the
reference duration `per`. With z standard normal, τ² is therefore the expected
total curvature measured in reference durations, independent of the time unit
and of the number of knots. An exponential prior on τ is the penalized-complexity
prior whose base model is a straight line; exponential(rate = 5.99) puts
P(τ > 0.5) = 0.05. With the level and slope priors above and γ = 0.125, a
prior-predictive calibration shows it keeps the 95% band of R(t) = β(t)/γ
inside [0.1, 20] while containing [0.5, 3], at every t, for the cubic knot sets
it covers (8, 12 and 20 uniform basis functions and 12 irregular; `make -f
tests/recovery/Makefile calibrate-smooth-prior`). Other priors, rates or
windows call for their own calibration.

The rules, checked at compile time: `smooth` is accepted only as the `coefs` of
a `bspline` of degree at least 2 (not of a `periodic_spline`); `modes` is the
bare name of an indexed parameter family with exactly `n_basis − 2` cells, each
declared `real` with prior exactly `normal(mu = 0.0, sigma = 1.0)` (**E414**
otherwise), since the whitening is what gives z that law; `per` is a duration
and a parameter in it is refused as structural (**E600**). Under `link = log`
the level and scale are dimensionless and the slope is per unit time (**E301**
otherwise). Other malformed uses are **E413**.

Declare the level, slope, scale and modes **without a range**, as above.
`fit run` gives a parameter with a finite search box (`in [lo, hi]` or
`[estimate]` `bounds`) a transform that keeps it in the box: for the level,
slope and modes the scaled logit, which would bend the whitened coordinates
the prior is built in; for the scale a log transform whose box ends are hard
walls, which the sampler's steps cross as divergences. Under `pgas`, `pmmh`
and `mh` it refuses such a box on any of the four, so the level, slope and
modes are sampled on the identity scale over (−∞, ∞) and the scale on the log
scale over (0, ∞). Their chains start from prior draws (`starts =
from_prior`, the default when every estimated parameter has a prior).

The gradients are emitted as for explicit coefficients: ∂g/∂a = 1, ∂g/∂b =
t_c − t_m, ∂g/∂zᵢ = (τ/κ)·Wᵢ(t_c) and ∂g/∂τ = Σᵢ zᵢWᵢ(t_c)/κ, where t_c is the
evaluation time clamped to the window and Wᵢ = Σⱼ Mⱼᵢ Bⱼ is the i-th whitened
basis function (each times f itself under `link = log`). Methods that read no
prior — `if2` and the NLopt optimizers — cannot estimate the scale beside the
modes, because the likelihood depends on τ and z only through w; `fit run`
refuses an estimated scale there and names the remedy (fix it, estimate
explicit coefficients, or fit with `pgas` or `pmmh`).

Shape errors are **E408** (both or neither knot form, a `window` that is not
`[a, b]`, `n_basis ≤ degree`, a degree above 5, a coefficient count that
disagrees with the knots), an unknown argument is **E409**, and an unknown
`link` is **E411**.

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

Anchors are accepted only in `breakpoints` (and a `bspline` `window`), not in
`values` or in any coefficient list: an anchor is a time, and the knots are the
only times a forcing declares. Same grammar as everywhere else — the anchor must be the whole
entry, the offset must carry a duration unit, and a `'months`/`'years` offset
under an `origin` is refused (see `value_at` above for the full rules and error
codes).

The forcing is then used exactly as any other — `C` or `C(t)` — and resolves to
`C(t − lag)`. `lag` is a property of the forcing as a whole, uniform across all
kinds (`interpolated`, `piecewise`, `periodic`, `sinusoidal`, `fourier`,
`periodic_spline`, `bspline`): every kind is evaluated at the single shifted
time.

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

An `interpolated` forcing is defined from its first data point to its last,
whatever its `method`. A run that evaluates it outside that range is an error
naming the forcing, its range and the side it was evaluated on, unless the run is
given `--allow-forcing-extrapolation` (in a fit config,
`[config] allow_forcing_extrapolation = true`), under which it holds its first
or last value there with one warning, as a `bspline` does outside its window.
`piecewise` and the periodic kinds are defined for all time.

#### Forcing data chosen when the model is run (`data = external("NAME")`)

`data = "path"` fixes the forcing's file in the model. `data = external("NAME")`
leaves it to be chosen when the model is run, exactly as for an
[`external("NAME")` table](#65-external-table-loading): the same model can then
be fitted under several versions of a covariate — a school calendar, a climate
series — without a model file per version.

```camdl preamble=forcing-demo
forcing {
  school_term : interpolated 'ratio {
    data      = external("school")   # the file is bound when the model is run
    time_col  = "date"
    value_col = "in_term"
    method    = constant
  }
}
```

The file is bound under the same name as a table's: `--table school=FILE`, or
`[data.tables] school = "FILE"` in a fit config, the command line winning where
both bind it. It is read exactly as a `data = "path"` file is read at compile
time — separator from the extension (`.csv` or `.tsv`), the first line that is
neither blank nor a `#` comment the header, `time_col` cells numbers in model
time or ISO dates under the model's `origin`, the rows the knots in file order,
the values scaled by the forcing's unit literal — and an indexed forcing reads
only the rows whose `key_col` holds its own level. The forcing is refused, naming
it and its external name, when the file is not bound, lacks a named column, has
a cell that is not a number (or, for `time_col`, a date), or has no row for one
of an indexed forcing's levels. The model's identity covers the declaration (the
name, the columns, the key and the unit); the file's bytes are an input of each
run, as for an external table.

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
hard-errors (`E377` — "compartment is stratified but used without indices"), so
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

A typed `let` takes the parameter's `dim` clause (§4.1.1), which is how a literal
coefficient with a compound dimension is written:
`let b : positive dim P^-1*T^-1 = 0.0001`. A unit literal in the body must agree
with the declared kind or clause: `let tau : rate = 30 'days` is **E282** (write
`let tau : duration = 30 'days`, or `let r : rate = 1 / (30 'days)`).

When a typed `let` has a constant body (a literal, a unit literal, or
arithmetic and pure math functions over them), the compiler evaluates it once
and emits it into the IR's `constants` section with its declared kind. Every use
reads it through a constant reference, so the dimensional analysis checker sees
the declared dimension rather than treating the constant as dimensionless, and
every gradient treats it as a constant, with derivative zero. It is not a
parameter: it cannot be fixed at another value, estimated, swept or sampled (§1,
doc comments).

Without the type annotation, a bare constant like `1e-6` is dimensionless and
adding it to a population compartment (`I + 1e-6`) triggers E302. The typed form
(`I + iota` where `iota : count`) resolves this because `iota` carries the
population dimension.

A typed `let` with a non-constant body is still inlined (or shared as a
binding) at each use, and its annotation is enforced there: every expanded body
must have the declared dimension, or it is **E318**, located at the `let`.
`let incidence : rate = gamma * I / N` is a checked assertion that the body is
a per-capita rate; `let flow : rate = gamma * I` is refused, since it is a
population rate (`positive dim P/T`). A `let` without an annotation takes its
body's dimension.

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

```camdl-doctest-preamble importation
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
```

```camdl preamble=importation
transitions {
  # Constant importation rate (exogenous FOI)
  importation[a in age, p in patch] : --> I[a, p]
    @ import_rate * age_weights[a] * patch_weights[p]
}
```

**Transition attributes: `#[lineage]`.** An attribute may precede a
transition's name, conventionally on its own line. `#[lineage]` is the only
attribute the language defines; anything else is **E110**.

```camdl-doctest-preamble lineage-sir
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
}
let N = S + I + R
```

```camdl preamble=lineage-sir
transitions {
  #[lineage]
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
```

The attribute marks the transition as a parent→child event, so the simulator can
record *which* individual infected *which* — an identity layer on top of the
compartment counts. It is for generating synthetic line lists and transmission
trees (for example, to validate phylodynamic inference against a known true
tree); it never changes the count dynamics, and the genealogy is a conditional
sample given the counts rather than a function of them. The three-stage
pipeline (`simulate --event-log`, `lineage realize`, `lineage tree`) is
described in [`docs/lineages.md`](lineages.md).

The rate must be **linear in the infector compartments**, because the per-parent
sampling weights are read off the rate as linear coefficients. `beta * S * I / N`
qualifies: `I` is a linear factor in the numerator, and its appearance in the
denominator `N` is treated as a frozen normalizer. A genuinely nonlinear use of
a parent compartment — `beta * S * (I + iota)^alpha / N` — is **E601**, which
names the offending compartment and the subexpression it appears in. So is a
product of two parent counts in the numerator, `beta * S * I * I / (N * N)` or
`beta * S * I * A / N`: its weights could not sum to the rate. A conditional
is linear when its condition reads no parent compartment and each branch is
linear: `if t > 10 then beta * S * I / N else 0` has the parent `I` with weight
`if t > 10 then beta * S / N else 0`. A condition that reads a parent anywhere,
`if I > 10 then …` or `if S / (K + I) > 0.5 then …`, is E601. Wrapping
the rate in `overdispersed(...)` (§9.8) is transparent to this analysis: the
environmental noise changes the counts, not the attribution.

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

**Repeated sources: several units per firing (`S + S --> D`).** A compartment
may appear more than once on the left. Its entries add, so `S + S --> D`
consumes two members of `S` each time it fires (stoichiometry `{S: -2, D: +1}`):
pair formation, or a reaction that needs two molecules of one species.

```camdl
time_unit = 'days
compartments { S, D, R }
parameters { k : rate in [0.0, 10.0]  g : rate in [0.0, 10.0] }
transitions {
  pair : S + S --> D  @ k * S * S / (S + D + R)
  rec  : S --> R      @ g * S
}
init { S = 5 }
simulate { from = 0 'days  to = 3 'days }
```

**Feasibility.** A transition is _feasible_ in a state when every compartment it
consumes holds at least the number of members one firing removes: `x_i ≥ ν_i`
for each compartment `i` whose net stoichiometry is `−ν_i < 0`. `pair` above is
feasible only while `S ≥ 2`; `rec` while `S ≥ 1`; `A + B --> C` while `A ≥ 1`
and `B ≥ 1`. The rule is about the net stoichiometry, so a catalyst (on both
sides, collapsed away as described above) is not consumed. How each backend
treats it:

- **Gillespie** fires a transition at its declared rate where it is feasible and
  not at all where it is not: the effective propensity is the rate expression
  when feasible and `0` otherwise. The rule applies to every transition, so no
  firing ever takes a compartment below zero. The rate expression is not
  rewritten — `k * S * S` stays `k * S * S` wherever `pair` can fire; it is not
  turned into the combinatorial `k * S * (S - 1)`. Write the number of distinct
  pairs into the rate if that is the intended propensity. For a single-unit
  transition whose rate is proportional to its source the rule changes nothing
  (the rate is already `0` when the source is empty); a constant-rate exit
  (`S --> R @ k`) stops when `S` reaches `0`.
- **Chain-binomial** refuses a transition that consumes more than one unit of a
  source, when the model is loaded and before any step, on `simulate` and on
  every inference method that runs on it (`pfilter`, `fit` with IF2, PMMH or
  PGAS). Its Euler-multinomial step draws how many _individuals_ leave a
  compartment, one at a time, and has no meaning for a firing that removes two
  at once:
  ```
  error: transition 'pair' consumes 2 units of S per firing; the chain-binomial backend models individuals leaving a compartment one at a time and supports at most 1 unit per firing from a source
    help: use --backend gillespie for this model (`fit` and `pfilter` have no Gillespie backend: fit it with backend = "ode", or restructure the transition)
  ```
- **ODE** treats every transition as a continuous flow: `pair` removes
  `S` at `2 × rate` and adds `D` at `rate`. There are no discrete firings, so
  there is no feasibility rule.

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
`probability` (dimensionless, domain `[0, 1]`). Because each branch fires at
`weight_i × rate`, the total outflow from the source is `rate × Σ weight_i`:
the rate after `@` is the outflow only when the weights sum to 1. A sum below 1
does not send the remainder anywhere; it lowers the outflow, and a sum above 1
raises it.

- **Constant weights must be a probability distribution.** A weight is a
  compile-time constant when it is a number (with or without a unit literal
  such as `'ratio`), arithmetic on such, a built-in math function (`exp`, `log`,
  `sqrt`, …) or `min`/`max`/`mod` applied to such, or an unindexed `let` binding
  of such. Each constant weight must be a finite number in `[0, 1]`, even
  beside parameter weights — `{ R : 1.5, Q : -0.5 }` sums to 1 but gives `Q` a
  negative rate — and when every weight is constant their sum must be within
  `1e-9` of 1. Otherwise the transition is rejected (**E423**). The tolerance
  absorbs floating-point rounding (`0.7 + 0.2 + 0.1` is `0.9999999999999999`;
  `1/3` three times) and nothing a model would write on purpose, so
  `{ A : 0.33, B : 0.33, C : 0.33 }` is an error. To lower the total outflow,
  change the rate after `@`.
- **Weights that depend on anything known only at run time are not checked.**
  A parameter, a table, a forcing or the state can make the sum differ from 1,
  and then the outflow is `rate × Σ weight_i`, exactly as written. Write the
  last weight as one minus the others (`{A : p, B : 1 - p}`, or
  `1 - sum-of-others` for n-way branching) and the sum is 1 for every value.

### 9.2 Indexed Transitions

```camdl
transition_name[i in dim1, j in dim2, ...] : from --> to  @ rate
```

The `[i in dim]` clause binds index variables. The compiler generates one
concrete IR transition per combination of index values. Dimensionality is known
at compile time: `|dim1| × |dim2| × ...` transitions.

**Every member must differ from the others.** Each generated transition is its
own flow, so two members with the same stoichiometry and the same rate run the
same flow twice. `inf[r in round] : S --> I @ beta * S` on unstratified `S` and
`I` never mentions `r`, so it would emit `inf_r1` and `inf_r2` and run at
`2 beta S` from a declaration that reads as `beta S`. A family whose expansion
yields two such members is **E422**, which names the binder that fails to
distinguish them. Index a compartment by it (`S[r] --> I[r]`) or a parameter
(`beta[r]`), or drop it. The rule compares the expanded members: a `where` guard
that leaves a single member is legal. A table lookup counts as using the binder
even when two entries are equal, because the members are compared before table
entries are folded to constants.

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

```camdl-doctest-preamble consecutive-aging
compartments { S, I, R }
dimensions {
  age   = [a0_5, a5_15, a15_50, a50_65, a65p]
  patch = [north, south]
}
tables { age_dur : age = [5.0, 10.0, 35.0, 15.0, 20.0] }
stratify(by = age)
stratify(by = patch)
```

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

```camdl-doctest-preamble erlang-E
compartments { E, I }
parameters { sigma : rate }
```

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

```camdl-doctest-preamble via-erlang
compartments { S, E, I, R }
parameters {
  beta  : rate
  sigma : rate
  gamma : rate
}
```

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

```camdl-doctest-preamble via-hyper-shared
time_unit = 'weeks
compartments { S, I, R }
parameters {
  beta    : rate
  p       : probability
  tau_typ : positive
  tau_pro : positive
}
```

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

```camdl-doctest-preamble via-hyper-dest
time_unit = 'days
compartments { S, E, I, R, D }
parameters {
  beta  : rate
  sigma : rate
  cfr   : probability
}
```

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

**Other exits from a staged compartment.** People can leave a compartment with
an Erlang dwell before the dwell completes: they die, are treated, or age into
another stratum. Write that as an ordinary `@` transition whose single source is
the staged compartment, indexed exactly as it was declared and with no stage
(`E[a]`, or bare `E` when `E` is not stratified). Such a transition is a
*competing exit*: a hazard that acts throughout the dwell and does not depend on
the stage.

```camdl-doctest-preamble via-competing
dimensions { age = [pre, sch] }
compartments { S, E, I, T }
stratify(by = age)
parameters {
  beta  : rate
  sigma : rate
  mu    : rate
  ar    : rate
  cmax  : rate
  K     : count
}
```

```camdl preamble=via-competing
transitions {
  infection[a in age] : S[a] --> E[a] @ beta * S[a]
  latency[a in age]   : E[a] --> I[a] via erlang(stages = 3, rate = sigma)
  death_E[a in age]   : E[a] --> @ mu * E[a]                          # leaves from any stage
  treat[a in age]     : E[a] --> T[a] @ cmax * E[a] * K / (K + E[a])  # saturating
  aging_E             : E[pre] --> E[sch] @ ar * E[pre]               # keeps the stage
}
```

The rate `R` you write is the total propensity out of the whole cell, as
everywhere in camdl, and `E[a]` in it is the sum over the stages. The compiler
expands the transition to one copy per stage and gives stage `s` the share of
`R` in proportion to its occupancy:

```
R_s = R * E_s / E     if E > 0
R_s = 0               if E = 0
```

Every stage then faces the same hazard per person, `R / E`, the stage rates sum
to `R`, and the `via` progression runs unchanged alongside. When the rate is a
product in which the source reference appears exactly once as a factor and the
compartment appears nowhere else — `mu * E[a]` — the share is that product with
the factor replaced by the stage cell, `mu * E[a, s]`, which is exactly what a
hand-written per-stage model holds. A quotient whose divisor does not mention
the compartment counts as such a product: `mu * E[a] / N_ref` gives
`mu * E[a, s] / N_ref`. A sum or difference of such products is replaced term
by term: `cmax * E[a] + mu * E[a]` gives `cmax * E[a, s] + mu * E[a, s]`, but
`cmax * E[a] + mu * K`, whose second term does not read the source, does not.
Any other rate (`treat` above) lowers to the guarded quotient.

Where the people go follows from the destination:

| Movement                                         | Stage                                 |
| ------------------------------------------------ | ------------------------------------- |
| Same compartment, different stratum (`aging_E`)  | Kept: `E[pre, s]` to `E[sch, s]`      |
| To an unstaged compartment (`death_E`, `treat`)  | Drawn from every stage                |
| To a *different* staged compartment              | Drawn from every stage, enter stage 1 |

A move into a different staged compartment restarts that compartment's dwell at
its first stage, as every inflow does. Because the modeller may have meant the
progression to continue (an isolation compartment that mirrors the infectious
period), the compiler says so with warning `W107`. The `via` transition's own
exit into a staged compartment does not warn. A `hyper_erlang` destination
warns too: its arrivals are split by the branch weights into each branch's
first stage. It has no stage dimension, so the way to continue into a branch is
to name that branch's stage compartment as the destination (`H__fast__2`),
which bypasses the branch weights.

*What `mean` and `rate` mean with a competing exit.* The `via` law's `mean` is
the dwell time absent the competing exit, not the mean time spent in the
compartment, which is shorter. With a constant hazard `mu` per person beside an
Erlang-`k` dwell of stage rate `k * sigma`, a person advances from each stage
with probability `k*sigma / (k*sigma + mu)`, so the fraction that completes the
dwell is

```
(k*sigma / (k*sigma + mu))^k
```

*Flows.* A flow reference to the transition as written reads all its stage
copies: `incidence(death_E[a])` is the deaths from `E[a]` at any stage. The
stage is not a population stratum, so this pools nothing you declared and `E280`
(§12.1) does not apply to it; a bare `incidence(death_E)` on the age-stratified
family is still `E280`, for pooling `age`. In trajectory and flow columns the
copies are named like any indexed family, with the stage level last
(`death_E_pre_s1`).

With `stages = 1` the source is not staged and the transition is an ordinary
second exit: the same model text compiles at every stage count.

*Lineage.* A `#[lineage]` competing exit takes the whole compartment cell as
its source, every stage of `E[a]`, as the unstaged transition takes `E[a]`. So
the stages are never parents of one another, and a rate is accepted or refused
(E601) at every stage count alike. Each stage copy's parent weight is its share
of the written rate's: for `mu * E[a] * I[a] / (K + E[a])` the parent of every
copy is `I[a]`, with weight `if E > 0 then mu * E / (K + E) * E_s / E else 0`,
and the weights of the copies sum to the unstaged transition's
`mu * E[a] / (K + E[a])`.

*Families over compartments.* A transition family bound over the compartment
set is a competing exit for each staged member and an ordinary exit for the
rest, so background mortality or ageing is written once for every compartment:

```camdl preamble=via-competing
transitions {
  infection[a in age] : S[a] --> E[a] @ beta * S[a]
  latency[a in age]   : E[a] --> I[a] via erlang(stages = 3, rate = sigma)
  bg[a in age, c in compartments] : c[a] --> @ mu * c[a]
  ageing[(a, a_next) in consecutive(age), c in compartments]
    : c[a] --> c[a_next] @ ar * c[a]
}
```

The members for `c = E` expand per stage (`bg_pre_E_s1`, …; `ageing` keeps the
stage), and `incidence(bg[a, E])` reads all of them. The refusals below apply
to the staged member. A `where` guard cannot name a compartment (`E217`), so a
family that must treat the staged member differently is written as separate
transitions instead.

Refused, each with its own diagnostic: a competing exit with
`overdispersed(...)` noise (`E361` — the noise multiplier is drawn per
transition, so the variance of the total would depend on the stage count); one
that consumes more than one individual of the staged compartment per event
(`E362`); a transition with two or more staged sources (`E363`); a staged
source together with an unstaged one (`E365`); and any `@` transition out of a
`hyper_erlang` compartment (`E364`), whose branches are separate chains rather
than one stage dimension.

**Scope and diagnostics.** A compartment takes one `via` law: a second `via`
transition on the same source is `E246`. The laws that ship are `erlang` and
`hyper_erlang`; any other law (`coxian`, `fixed`, …) is `E243`. `hyper_erlang`
on an already-stratified source is a later sub-phase, rejected today with `E248`.
Argument mistakes each get a named code: a `stages` that is not a positive
integer is `E244`; giving both or neither of `mean`/`rate` is `E245`; an unknown
`erlang` keyword is `E247`; fewer than two branches is `E255`; a `weight` on the
last branch (or a missing one on any earlier branch) is `E256`; and duplicate
branch labels are `E258`.

**Moving people into and out of a staged compartment.** `init`, inflow
transitions (`--> E`), `transfer` and `add` treat a compartment with an Erlang
dwell like any other: write the compartment, indexed exactly as it was declared
and with no stage (`E[a]`, or bare `E` when `E` is not stratified), and the
compiler decides which stages the people come from and go to. This holds in
`interventions {}`, `events {}` and `reactive_interventions {}` alike (§13).

```camdl preamble=via-competing
transitions {
  infection[a in age] : S[a] --> E[a] @ beta * S[a]
  latency[a in age]   : E[a] --> I[a] via erlang(stages = 3, rate = sigma)
}
interventions {
  promote : transfer(fraction = 0.2, from = E[pre], to = E[sch]) at [30 'days]  # keeps the stage
  expose  : transfer(fraction = 0.1, from = S[pre], to = E[pre]) at [40 'days]  # enters stage 1
  treat   : transfer(fraction = 0.5, from = E, to = T)           at [50 'days]  # from every stage, per age
  remove[a in age] : transfer(count = 3, from = E[a], to = T[a]) at [60 'days]
  import  : add(E[pre], 5)                                       at [70 'days]  # enters stage 1
}
```

| Movement                                         | Stage                                 |
| ------------------------------------------------ | ------------------------------------- |
| `init`, an inflow transition, `add`              | Enter stage 1                         |
| `transfer` from an unstaged compartment          | Enter stage 1                         |
| `transfer` within the compartment (`promote`)    | Kept: `E[pre, s]` to `E[sch, s]`      |
| `transfer` to an unstaged compartment (`treat`)  | Drawn from every stage                |
| `transfer` to a *different* staged compartment   | Drawn from every stage, enter stage 1 |

*One amount for the whole compartment.* A `transfer` out of a staged compartment
is one action over all its stages (a *grouped transfer*). Its `fraction` or
`count` is evaluated once and applies to the compartment's total `N`, the sum
of the stages. On integer compartments the number moved is `floor(N * fraction)`
or `min(round(count), N)`, and it is then taken from the stages in proportion
to their occupancy: stage `i` gives `floor(M * n_i / N)`, and the individuals
left over go one each to the stages with the largest remainders, the earlier
stage first on a tie. So 5 people in stages (3, 1, 1) at `fraction = 0.4` give
up `floor(5 * 0.4) = 2`, as (1, 1, 0). No stage gives more than it holds. The
number moved therefore does not depend on `stages`, and the same text compiles
at `stages = 1`, where it is an ordinary transfer.

The floor is applied once, to the compartment total. This is a deliberate
restriction of what a staged `transfer` can express: it cannot floor each
stage separately. A transfer that names its cells one at a time does that, and
is unchanged: a transfer between unstaged compartments, or one written with an
explicit stage on its source (`from = stage(E[pre], 2)`), still moves
`floor(source * fraction)` of that one cell. On real compartments, and on the
ODE backend, nothing is rounded: each stage gives `n_i * fraction`, or
`min(count, N) * n_i / N`.

*Population dimensions are never pooled.* `transfer(from = E, to = T)` with
both stratified by `age` pairs cell for cell over `age`, as for any bare
stratified transfer (§13.1), and groups only the stages within each age cell.
`count` on such a bare transfer is still `E238`, and a bare staged endpoint
inside an indexed family is still `E239`.

*What is refused.* A `set` on the compartment (`{ E[pre] = 5 ... }`) is `E366`:
a total has no single meaning across the stages; set one stage
(`stage(E[pre], 1) = ...`). A source that names its stage with a destination in
the same compartment that does not (`from = stage(E[pre], 2), to = E[sch]`) is
`E367`: name the destination stage as well. A `transfer`, `add` or `set` on a
`hyper_erlang` compartment is `E368`. A transfer from one staged compartment
into a different one restarts the destination's dwell and warns (`W107`), as the
corresponding transition does; a movement with either endpoint written as
`stage(...)` does not warn.

Reads are unaffected: a bare `E`, or `E[a]`, sums every stage (§5.1).

**Naming one stage.** Everything above writes the compartment and leaves the
stages to the compiler. Two forms name a stage when the model means one:

```camdl preamble=via-competing
transitions {
  infection[a in age] : S[a] --> E[a] @ beta * S[a]
  latency[a in age]   : E[a] --> I[a] via erlang(stages = 3, rate = sigma)
  late_death : stage(E[pre], 3) --> @ mu * stage(E[pre], 3)       # out of stage 3 only
}
interventions {
  treat_late : transfer(fraction = 0.8, from = stage(E[pre], 3), to = T[pre]) at [30 'days]
  move[st in stages(E)] : transfer(fraction = 0.2, from = stage(E[pre], st), to = stage(E[sch], st)) at [40 'days]
  import : add(stage(E[pre], 2), 5) at [50 'days]
  reset  : { stage(E[sch], 1) = 0  at = [60 'days] }
}
init { S[pre] = 990  stage(E[pre], 2) = 10 }
```

- `stage(E[pre], 2)` is one residence location: stage 2 of `E[pre]`. Its first
  argument is the compartment with every population dimension indexed and no
  stage (bare `E` when `E` is not stratified); its second is the location, a
  whole number from 1 to the compartment's `stages`, or a variable bound by
  `in stages(E)`. It is a state expression (in a rate, a `let`, an observation,
  `balance`: `beta * S * stage(I, 2) / N` is infectiousness in stage 2 only) and
  an endpoint (a transition source or destination, `from` and `to` of a
  `transfer`, the target of `add` and `set`, an `init` entry).
- `stages(E)` is the set of residence locations of `E`, and is written only as
  the domain of a binder: `[st in stages(E)]` on a transition, an intervention,
  an event, a reactive intervention, a quantity or a `let`, and
  `sum(st in stages(E), ...)`. The
  members of such a family are named with the stage level last (`move_s1`,
  `move_s2`, `move_s3`), and a flow reference to a transition family over
  stages reads all of them, as it does for a competing exit.

A `stage(...)` endpoint is that location only. A transition whose source is
`stage(E[pre], 3)` is an ordinary transition out of that one cell, not a
competing exit: its rate is not divided over the stages. An `init` entry or an
`add` without a stage lands in stage 1; with `stage(...)` it lands where it
says. A movement with either endpoint written as `stage(...)` states its
location and does not raise `W107`.

A stage is a stochastic progression state, not an elapsed-time interval:
`stage(E, 3)` selects people who have completed two stage transitions, not
people infected between two dates.

*The same text at every stage count.* With `stages = 1`, and for a compartment
with no `via` law, there is one location: `stage(E[pre], 1)` is `E[pre]`, and a
binder over `stages(E)` has one element, so its declaration has the single
member it would have without the binder (`move`, not `move_s1`). A model
written with location 1 and stage variables compiles unchanged when `stages`
changes; a literal location above the stage count does not. There is no
stage-count accessor: an amount written for the whole compartment is divided
over its stages by the compiler, so a fraction or a rate never needs the number
of stages.

*What is refused.* `stage(...)` with anything but a compartment reference and a
location, or with a population dimension left unindexed (`stage(E, 2)` when `E`
is stratified by `age`), is `E369`. A location the compartment does not have
is `E370`, and the message gives its stage count: a literal out of range, any
location but 1 on a compartment with one stage, or a location that is neither a
literal nor a stage variable. A variable bound over `stages(E)` is a location
of `E` only: as the location of another compartment (`stage(I[a], st)`) it is
`E371`, and anywhere else (as a population index, `S[st]`; as a value; in a
`where` predicate) it is `E372`. The one other thing it indexes is a `let`
family declared over `stages(E)`. `stages(...)` of something that is not a
compartment, or written as a value, is `E374`, as is a binder over it on an
observation stream (data has no column for a stage) or in the brackets of an
`init` entry. On a `hyper_erlang` compartment
a location would be a branch and a phase; both forms are reserved there and
are `E373`.

The stage dimension the compiler generates (`__latency_stage`, with levels
`s1`, `s2`, …) is not part of the language. Naming it (`[st in
__latency_stage]`, `sum(s in __latency_stage, ...)`) is `E375`, and indexing a
stage by its level (`E[pre, s2]`) is `E376`; each diagnostic gives the
`stages(E)` or `stage(E[pre], 2)` to write instead. An expanded cell name
(`E_pre_s2`) is how every stratified cell is named in trajectory columns, and
may still be written wherever a cell name is accepted; it does not exist at
`stages = 1`.

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

```camdl-doctest-preamble compartment-iter
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
```

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

```camdl-doctest-preamble table-matrices
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
```

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

```camdl-doctest-preamble ode-demo
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
```

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

An observation stream states how one column of data relates to the model: the
model quantity the column measures, called the projection, and the
distribution of a recorded value around it. Simulation draws from that
distribution and inference scores data under it (§12.4). The data-file format
is specified in `camdl docs data`; fitting is described in
`camdl docs inference`.

```camdl-doctest-preamble obs-sir
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
  pi_zero  : probability
}
let Ntot = S + I + R
transitions {
  infection : S --> I @ beta * S * I / Ntot
  recovery  : I --> R @ gamma * I
}
```

```camdl preamble=obs-sir
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }

  detection {
    columns       { time : time, detection : count }
    projected     = I
    emit_schedule = every 14 'days
    detection     ~ bernoulli(p = p_detect * projected / N)
  }
}
```

A stream is written `name { … }`, or `name from <source> { … }` to name its
data source; without `from`, the source is the stream's own name. The block
takes five keys, each at most once:

| key                       | states                                                                                                                                                   |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `columns { … }`           | the data file's columns, by name and role: exactly one set of temporal columns (a `: time` column, or a `window_start`/`window_stop` pair, §12.1.1) and the value column |
| `covers = …`              | the period each row covers, on a stream whose projection accumulates a flow (§12.1.1)                                                                    |
| `projected = …`           | the projection (§12.1); the measurement line reads its value as `projected`                                                                              |
| `<value_col> ~ FAMILY(…)` | the measurement: the distribution of the value column, with named arguments (§12.2)                                                                      |
| `emit_schedule = …`       | the times at which simulation emits an observation (§12.4); a model that is only fitted may omit it                                                     |

Any other key assigned a value (`foo = incidence(…)`, `projeted = …`) is
**E121**. Its fix-it renames the key to `projected` when the block has no
`projected` line; otherwise its hint lists the accepted keys. A key given twice
is **E122**. A colon after the stream name (`name : { … }`) is **E270**;
`likelihood = FAMILY(…)` is **E273**, written `<value_col> ~ FAMILY(…)`; and a
prior's `| dim` pooling suffix on the measurement line is **E271**, since a
stream is stratified by indexing its header (§12.3).

Related sections: §25.4 (how projections lower to the IR), §26 (the error
table), §21 (the CLI that simulates and fits).

### 12.1 Projections

A projection is the model quantity that one row of a stream measures: a flow
accumulated over the period the row covers, or a stock read at the row's time.

```camdl
incidence(transition)                      flow accumulated over the period the row covers (§12.1.1)
incidence(transition[north])               positional index, in declaration order
incidence(transition[patch = north])       named index, in any order
compartment                                current population, as a count
compartment[age = child]                   named index on a compartment
prevalence(of = I[a], among = N_local[a])  a proportion, of / among, with checks
```

**Indexing.** An index on a projection selects one cell of a stratified family;
it does not sum over the dimensions it leaves out. Every declared dimension is
given, either all by position in declaration order or all by name in any
order; mixing the two styles in one reference is **E326**. On a `[patch, age]`
family, `infection[patch = north, age = child]` names `infection_north_child`.
A partial index names no cell and is refused: `incidence(infection[patch =
north])` is **E507**, an unknown transition (there is no `infection_north`),
and `I[child]` on an `I` stratified by age and patch is **E287**. A sum over a
dimension is written out (see "Pooling strata" below).

A named index whose label is not a dimension of the family is **E332**. A
positional index is matched to a slot by position and checked only against that
slot's levels, so when two dimensions share level names, a swapped pair names a
different existing cell and compiles without a diagnostic (§1). A named index
does not have this failure and keeps its meaning when the dimension
declarations are reordered, so it is the form to use wherever two dimensions
could share a level name.

**Flows.** `incidence(tr)` is the number of `tr` events in the period the row
covers (§12.1.1). It takes exactly one transition: several is **E203** and none
is **E250**. It may appear only as the whole right-hand side of
`projected = …`, under `sum(…)`, and as a term of a sum of such terms; it is not
an expression function. Two flows observed as one column are added,
`incidence(a) + incidence(b)`, and the strata of one family are pooled with
`sum(a in age, incidence(tr[a]))`. The two forms combine into one list of
flows (§25.4). A `where` guard on the sum prunes its domain as it does in a rate
expression: `sum(p in patch where p != north, incidence(infection[p]))` pools
the remaining levels.

A projection whose leaves are all flows accumulates over the row's period; one
whose leaves are all states is read at an instant. Mixing the two is **E341**,
and so is any operation on flows other than addition and the one ratio below:
weighting a term (`rho * incidence(a)`, `rho * sum(p in patch,
incidence(infection[p]))`), subtracting one, dividing by a constant or a state,
or dividing twice. A per-stream reporting rate belongs in the measurement:
`cases ~ poisson(rate = rho * projected)`.

**Ratios of flows.** One sum of flows may be divided by another:
`incidence(a) / (incidence(a) + incidence(b))` is the fraction of the period's
events that were `a` events, both sums accumulating over the same period. A
side whose `where` guard prunes every level is **E351**. The ratio is
dimensionless, so it fits a probability argument (`binomial`'s `p`,
`beta_binomial`'s `mean`, `beta`'s `mean`, `bernoulli`'s `p`), and a count
family (`poisson`, `neg_binomial`) refuses it with **E304**.

When no denominator event occurs in a period the ratio is undefined and the
projection is `NaN`. Under `binomial` and `beta_binomial` a row with `n = 0`
then scores exactly 0 and a row with `n > 0` is refused, with the argument
named; under `beta` and `bernoulli` the row is refused. The numerator need not
count a subset of the denominator's events: `incidence(die) /
incidence(onset)` is the case fatality within a period and exceeds 1 on a
trajectory whose deaths outnumber its onsets, which a binomial scores as
impossible.

**Ratios of rates.** A projection that divides an expression at least one of
whose additive terms is structurally equal to a transition's rate, such as
`mu_c * I / (mu_c * I + mu_f * H)`, is **E352**. That expression is read at the
row's time and gives the share of the hazard at that instant, which equals the
share of the period's events only when both flows are constant over the
period. The hint writes both alternatives: the ratio of flows, for the share of
the period's events, and `prevalence(of = <numerator>, among = <denominator>)`,
for the instantaneous ratio of hazards. The comparison is structural, so a rate
written differently in the projection and in the transition is not matched: the
check has no false positives and misses some cases.

**Stocks and proportions.** A compartment expression in `projected` is a count:
on an age-stratified model, `projected = I[child]` is the number of infected
children. A proportion names its denominator: `prevalence(of = I[child], among
= N_local[child])` is the share of children who are infected. It has the same
value and the same IR as `I[child] / N_local[child]`, and adds three checks that
the plain division, which remains legal, does not have:

- every compartment cell `of` reads is also read by `among` (**E357**), so
  `among = S[child] + R[child]` and `among = N_local[adult]` are refused against
  `of = I[child]`. The check passes trivially when `among` reads no cell (a
  parameter, a constant or a table, all valid denominators). It does not bound
  the ratio by 1 (`of = I[child] + I[child]` passes), and it does not apply to a
  denominator written as an explicit list of cells instead of an indexed family;
- both sides collapse the same dimensions (**E358**): `of = sum(a in age, I[a,
  p])` against `among = sum(a in age, sum(q in patch, N[a, q]))` pools age in
  the numerator and age and patch in the denominator, and is wrong by about the
  number of patches;
- neither side reads a flow (**E359**), since a proportion of a stock is read at
  an instant and `incidence(…)` accumulates over a period.

`prevalence` has no pooling argument of its own; pooling is written with `sum`
on each side. It is valid only as the whole right-hand side of
`projected = …`. Inside arithmetic it is refused with **E100**, whose message
says where `prevalence` is accepted and, in a projection, whose hint writes the
expression out with each call replaced by its division. A proportion inside
arithmetic is written as the division,
`sum(a in age, w[a] * (I[a] / N_local[a]))`. A missing `of` or `among` is
**E250**. `prevalence(X)` and `prevalence(X1, X2)` are **E356**: write `X` for
the count, or `prevalence(of = X, among = N)` for the proportion. A proportion
is dimensionless, so it fits a probability argument
(`binomial(n = tested, p = projected)`), and a count family refuses it with
**E304**.

**Other expressions.** `projected` accepts any expression over compartment
state, parameters and time. A count over several compartments is their sum,
`projected = x3 + y3`, and a proportion may be written as a plain division:

```camdl-doctest-preamble obs-derived
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
```

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

Simulation and inference evaluate a projection identically.

**Pooling strata.** On a stratified family, an unindexed `incidence(infection)`
is **E280**, so that pooling across strata is always written out; the
diagnostic prints the two forms below with the family's own dimensions. A
pooled column with one reporting rate sums the family:

```camdl
projected = sum(p in patch, incidence(infection[p]))
cases ~ poisson(rate = rho * projected)
```

A family stratified over several dimensions takes one `sum` per dimension:
`sum(a in age, sum(p in patch, incidence(infection[a, p])))`. One row per
stratum, each with its own reporting rate, is an indexed stream (§12.3):

```camdl
cases[p in patch] {
  columns   { time : time, patch : dim, cases : count }
  covers    = closing_at(time, 7 'days)
  projected = incidence(infection[p])
  cases ~ poisson(rate = rho[p] * projected)
}
```

On an indexed stream an unindexed `incidence(infection)` would score every row
against the pooled total; it is **E280** there too, and the hint names the
stream's binder (`incidence(infection[p])`).

The same rule applies to state. A compartment written unindexed in `projected`
over a population stratum is **E280**: `projected = I` on an age-stratified `I`
assumes one reporting rate across age groups, and a rate per stratum is a
different model. The pooling is stated with `sum(I)` for the total,
`sum(a in age, I[a])` to reduce each dimension by name, or an indexed stream
(`prev[a in age] { … projected = I[a] … }`). An unindexed name inside a binder
`sum` is not stated: `sum(a in age, I[a] + R)` still pools `R`, and is
**E280**. A `let` carries the same rule: `let N = S + I + R` read by a
projection is **E280** at the use, and `let N = sum(S + I + R)`, or `sum(N)` at
the use, states the pooling.

Residence structure is exempt. On a compartment staged by `via erlang`, `I`
pools only its stages; `I[child]` on an age-stratified staged `I` pools that
stratum's stages; and `incidence(death[child])` on a transition out of a staged
compartment reads that stratum's stage copies (§9.4.1). Staging declared by
hand in `dimensions {}` is a population stratum and is not exempt. The rule
applies only to observations: in model dynamics, such as a force of infection,
an unindexed `I` is the total by definition.

Within the measurement line, `projected` is the stream's evaluated projection.

### 12.1.1 What a row covers: `covers` and the window columns

An `incidence(...)` projection accumulates a flow over an interval, and a row's
time label alone does not determine the interval. `2026-07-08` is the day 8 July
in a daily file labelled by the day the count describes, the week 8–14 July in
a week-starting file, the week 2–8 July in a week-ending one, and the week 1–7
July in a file whose label is the boundary at which its window closes, which is
the convention of pomp's accumulator variables and of `camdl simulate --obs`
output. A stream whose `projected` accumulates a flow (`incidence(...)`,
`sum(..., incidence(...))`, added incidence terms, or a ratio of two such sums,
§12.1) therefore states what each of its rows covers. Declaring neither
`covers = …` nor `window_start`/`window_stop` columns is **E350**; there is no
default. A stream whose `projected` reads state at an instant (a compartment
expression, or `prevalence(of = …, among = …)`) has no window, and declaring one
by either means is **E348**.

**Forms.** Four forms take the stream's own `: time` column and a rule; the
fifth puts both boundaries in the file. With `D` the label in a row's time
column, resolved for `D = 11 July` and a seven-day width:

| form                                   | the row covers          | `D = 11 Jul`, `7 'days`               |
| -------------------------------------- | ----------------------- | ------------------------------------- |
| `covers = day(time)`                   | `[D, D+1 day)`          | `[11 Jul, 12 Jul)`                    |
| `covers = starting_on(time, 7 'days)`  | `[D, D+7 days)`         | `[11 Jul, 18 Jul)`                    |
| `covers = ending_on(time, 7 'days)`    | `[D−6 days, D+1 day)`   | `[5 Jul, 12 Jul)` — 11th **included** |
| `covers = closing_at(time, 7 'days)`   | `[D−7 days, D)`         | `[4 Jul, 11 Jul)` — 11th **excluded** |
| `window_start` + `window_stop` columns | exactly `[start, stop)` | as written                            |

`ending_on` and `closing_at` differ by a whole day. "Week ending 11 July" means
11 July is the **last included day**: under `ending_on` the span opens six days
before the label and closes one day after it. Under `closing_at` the label is
the instant at which the window **closes**, and the label itself belongs to the
next row's window. That is the reading of pomp's accumulator variables (the
value reported at `t[k+1]` is the flow since `t[k]`) and of every file
`simulate --obs` writes, so synthetic data is read back correctly under
`closing_at(time, <cadence>)` with the cadence of the stream's `emit_schedule`.
A file typed from a surveillance bulletin is labelled by the day or week it
reports on and takes `day`, `starting_on` or `ending_on`.

The width is a compile-time constant duration; `day(...)` takes none. A width
given to `day`, a width missing from the other three, an unknown form name, or a
width naming a data column is **E349**; a width that varies from row to row is
stated with the window columns. Under `origin = date(...)` a calendar width
(`1 'months`, `1 'years`) is **E321**: the width is one fixed span for every
row and a calendar month is 28 to 31 days, so monthly or yearly periods are
stated with the window columns. `covers` names the stream's own `: time`
column; any other name is **E347**.

**Window columns.** `window_start` and `window_stop` are `columns { }` roles
alongside `time`, `dim` and `count`. They replace the `: time` column and are
the whole declaration: a stream carrying them has no `covers` line. Both a
`: time` column and a window pair, half a pair, or a pair together with `covers`
is **E347**. Each row covers exactly `[start, stop)` as written, so this is the
only form that can state a width that changes from row to row (a fortnightly
row in a weekly file, after a suspended publication) or a span that no row
covers. The stop is the stream's time in a fit: output rows, `--score-from` and
a forecast grid all read the closing boundary, so a windowed stream sits on the
time axis where a `: time`-column stream with the same closing boundary does.

```camdl preamble=obs-sir
observations {
  cases {
    columns   { onset_from : window_start, onset_stop : window_stop,
                cases : count }
    projected = incidence(infection)
    cases     ~ neg_binomial(mean = rho * projected, r = k)
  }
}
```

**Gaps and overlaps.** Under the four uniform forms every row's window follows
from its label, so consecutive rows whose windows do not touch mean a row is
missing. That is refused when the data is bound, with an error naming both
windows and the uncovered span between them. Every scheduled row is kept, and an
unobserved one is written as `NA`: a hole, with no likelihood term, whose window
is kept. Under the window columns a gap is legal: the flow over the uncovered
span belongs to no row and is discarded, and the next row's accumulation opens
at its own `window_start`. Overlapping windows are refused under every form.

**Where scoring begins.** The process is simulated from `simulate.from`, and
nothing before the first period's start is scored, so the first declared period
is the end of the warm-up. A first period that opens before `simulate.from` is
an error naming both times, because that time is never simulated and the row
would be scored against flow that does not exist. A first period opening
exactly at `simulate.from` is accepted.

The data-file side (file layouts, `NA` holes, and what the declaration changes
for an existing file) is specified in `camdl docs data` ("What a row covers").

### 12.2 Likelihood Families

The measurement line `<value_col> ~ FAMILY(…)` gives the distribution of the
value column around the projection. Each family takes named arguments, in which
`projected` is the stream's projection.

```camdl
neg_binomial(mean = EXPR, r = EXPR)            overdispersed counts
poisson(rate = EXPR)                           Poisson counts
normal(mean = EXPR, sd = EXPR)                 counts, as a rounded normal
binomial(n = EXPR, p = EXPR)                   bounded counts
beta_binomial(n = EXPR, alpha = EXPR, beta = EXPR)          overdispersed prevalence (raw)
beta_binomial(n = EXPR, mean = EXPR, concentration = EXPR)  overdispersed prevalence (mean/concentration)
beta(mean = EXPR, concentration = EXPR)        continuous proportion in (0, 1)
bernoulli(p = EXPR)                            binary outcome
zero_inflated(base = neg_binomial(mean = EXPR, r = EXPR), pi = EXPR)
                                               counts with excess zeros
```

`neg_binomial(mean = μ, r = k)` is the **NB2** (mean–dispersion)
parameterization: the mean is `μ` and the variance is `μ + μ²/r`, so a smaller
`r` means more overdispersion and `r → ∞` recovers `poisson(μ)`. A prior on `r`
is therefore a prior on the quadratic excess variance `μ²/r`.

`normal(mean = μ, sd = σ)` is a **discretized normal for counts** (He et al.
2010), not a continuous density. Inference scores an observed count `y` as the
probability that a normal variable rounds to it,
`P(y − 0.5 < X < y + 0.5)` with `X ~ Normal(μ, σ²)`, and `y = 0` as
`P(X < 0.5)`; an observation is rounded to the nearest non-negative integer
before scoring, and the probability is floored at `1e-18`. Simulated
observations are a normal draw rounded to the nearest integer and floored at
0. A non-integer observation is scored as its rounded value, with a run note
saying so once.

**Count arguments.** `neg_binomial`'s `mean`, `poisson`'s `rate`, and
`binomial`'s and `beta_binomial`'s `n` carry the count dimension `[P]`: each is
an expected count over the reporting interval, not a per-time rate. A
transition rate there (`projected = gamma * I`, dimension `P·T⁻¹`) is
**E304**. At a one-day reporting step a per-day rate and a one-day accumulated
count coincide numerically; at a weekly step the likelihood would be wrong by
about the window length. `incidence(<transition>)` accumulates the flow over the
interval. A bare numeric literal (`mean = 100`) is a count by context and is
exempt.

**Proportion arguments.** `binomial`'s `p`, `beta`'s `mean`, `bernoulli`'s `p`
and `beta_binomial`'s `mean` are dimensionless fractions, fed by a proportion
(`I / N`, `prevalence(of = …, among = …)`) or a ratio of two accumulated flows
(`incidence(a) / (incidence(a) + incidence(b))`, §12.1). A count there
(`p = projected` over a bare `incidence(a)`, a missing division by `N`) is
**E304**, and so is a fraction given to a count argument.

The two `beta_binomial` forms are equivalent: `mean`/`concentration` lowers to
`alpha = mean · concentration`, `beta = (1 − mean) · concentration`. Mixing the
two forms in one call is **E252**. A parameterization of `concentration` that
samples well is described in `camdl docs inference` (reparameterization).

`beta(...)` is for an observed value that is itself a **continuous
proportion** in the open interval (0, 1), such as a coverage or a positivity
reported as a fraction; a `k`-of-`n` count is `beta_binomial`. It uses the same
mapping, `alpha = mean · concentration`, `beta = (1 − mean) · concentration`.
`mean` and `concentration` are both differentiable, so `beta` is accepted by
the gradient-based methods as well as the others.

### 12.2.1 Imperfect tests: `diagnostic_test`

`diagnostic_test` scores the positives of an imperfect test, such as slide
microscopy, a rapid diagnostic test (RDT) or PCR, whose sensitivity and
specificity are below 1. It wraps a probability family and rewrites its
probability argument:

```camdl preamble=obs-sir
observations {
  slide_positivity {
    columns          { time : time, slide_positivity : count }
    projected        = I
    emit_schedule    = every 1 'weeks
    slide_positivity ~ diagnostic_test(
      base = binomial(n = N_tested, p = projected / N),
      sens = rho_sens,
      spec = rho_spec
    )
  }
}
```

**Meaning.** The rewrite happens at compile time. With true positive fraction
π, the probability of a positive test result is

```
p_observed  =  sens · π  +  (1 − spec) · (1 − π)
```

The first term is the true positives (infected and detected), the second the
false positives (uninfected and reported positive). The compiler replaces the
base's `p = π` with this expression, and the IR is identical to that of the
measurement written out by hand:

```camdl
slide_positivity ~ binomial(
  n = N_tested,
  p = rho_sens * (projected / N) + (1 - rho_spec) * (1 - projected / N)
)
```

**Bases.** The base is one of two families:

- `binomial(n = …, p = …)`, a survey of `n` individuals counting positives;
- `bernoulli(p = …)`, one test with a 0/1 result.

Sensitivity and specificity correct a probability, so any other base
(`poisson`, `neg_binomial`, `normal`, `beta_binomial`) is **E253**. A call
missing `base`, `sens` or `spec` is **E254**.

**Arguments.** `sens` and `spec` may be constants, parameters with priors
(estimating the test characteristics jointly with the transmission model), or
expressions. The compiler checks that each is dimensionless (**E304**
otherwise). A `sens` or `spec` that is a compile-time constant (a number,
arithmetic on numbers, or an unindexed `let` of one, as for the weights of a
destination split, §9.1.2) must be a finite number in `[0, 1]`: `sens = 1.5`
or `spec = 1 - 1.2` is **E379**. A parameter is not checked when the model is
compiled; declaring it `probability` keeps its values in `[0, 1]` (§4.4).

### 12.2.2 Counts with excess zeros: `zero_inflated`

`zero_inflated` scores counts with more zeros than a negative binomial
produces, where the extra zeros come from a process separate from the
transmission dynamics (a trapping site where the vector is absent, a week with
no sampling effort). These are called **structural zeros**. Under a plain
`neg_binomial` the one dispersion parameter `r` has to account for both the
zeros and the spread of the positive counts, which lowers `r`, overstates the
overdispersion of the positive counts and biases the fitted reporting rate.
`zero_inflated` gives the structural zeros their own probability `pi`, mixing a
point mass at zero with a negative binomial base:

```
P(y = 0)  =  pi  +  (1 - pi) * f_NB(0 | mean, r)
P(y = k)  =        (1 - pi) * f_NB(k | mean, r)      for k > 0
```

`pi` is the probability that a row is a structural zero. At `pi = 0` the family
is exactly the plain `neg_binomial`, so the posterior of `pi` measures the share
of zeros that the dynamics do not account for.

```camdl preamble=obs-sir
observations {
  vector_catch {
    columns       { time : time, vector_catch : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    vector_catch  ~ zero_inflated(
      base = neg_binomial(mean = rho * projected, r = k),
      pi   = pi_zero
    )
  }
}
```

**Arguments.** `base` is `neg_binomial(...)`; any other family is **E324**.
Both `base` and `pi` are required (**E325**), and any other keyword is
**E251**.

**Dimensions.** `mean` carries the count dimension `[P]`, an expected count over
the reporting interval as in the plain `neg_binomial`, so a per-time transition
rate there is **E304**. `r` and `pi` are dimensionless. A `pi` outside
`[0, 1]` scores the row as impossible (log-likelihood −∞). Declaring `pi`'s
parameter as `probability` checks its `[0, 1]` domain when parameter values are
resolved and restricts inference to `[0, 1]` (§4.4).

**Gradients.** All three arguments are differentiable, so the compiler emits a
gradient for each and every inference method accepts the family. Write
`f₀ = f_NB(0 | mean, r)` and `S = pi + (1 − pi)·f₀`, and let `w = pi/S` be the
posterior probability that an observed zero is structural (`w = 0` whenever the
count is positive). The derivatives in `mean` and `r` are `(1 − w)` times the
negative binomial ones. The derivative in `pi` is `(1 − f₀)/S` at zero and
`−1/(1 − pi)` above it.

As `pi → 0` the derivative in `pi` at an observed zero grows like `1/f₀`. A
`probability` parameter is sampled on the logit scale, where `pi = 0` lies at
−∞ and is never evaluated, and the chain rule multiplies that derivative by
`pi·(1 − pi)`, which goes to 0 at the boundary. Declaring `pi` as `probability`
is therefore enough to keep the gradient bounded near `pi = 0`.

Stan handles the boundary the same way. It has no built-in zero-inflated family
either; its documented idiom declares the mixing probability
`real<lower=0, upper=1>` and scores the zero case with `log_sum_exp`, relying on
the automatic constraint transform rather than on anything in the density (Stan
Development Team, _Stan User's Guide_ v2.39, ["Zero-inflated and hurdle
models"](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#zero-inflated)).

### 12.3 Indexed Observations

An indexed stream observes each stratum of a dimension in its own stream. The
header `name[p in dim]` binds `p` to each level of `dim` in turn, and `p` may
index both the projection and the measurement's arguments.

```camdl-doctest-preamble obs-patch
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
```

```camdl preamble=obs-patch
observations {
  cases_by_patch[p in patch] {
    columns        { time : time, patch : dim, cases_by_patch : count }
    covers         = closing_at(time, 7 'days)
    projected      = incidence(infection[patch = p])
    emit_schedule  = every 7 'days
    cases_by_patch ~ neg_binomial(mean = rho * projected, r = k)
  }
}
```

This declares three streams, `cases_by_patch_north`, `cases_by_patch_south`
and `cases_by_patch_east`, each with its own projection
`incidence(infection[patch = north])` and so on. A parameter indexed by the
same dimension is read per stream: `rho[p]` in the measurement line is `rho_north`
in the first stream.

### 12.4 Simulation and inference

Each observation stream is read by both simulation and inference.

- **Simulation** (`camdl simulate`). At each time in the stream's
  `emit_schedule`, the runtime evaluates the stream's projection and draws an
  observation from the declared distribution at that value. The draws are
  written only when `--obs`, `--obs-dir` or `--obs-only` is passed (§21). To
  simulate on the times of a data set instead, pass `--design-from fit.toml`
  (run spec §4.6).
- **Inference** (`camdl fit`). At each time in the data, the runtime evaluates
  the same projection and adds the log density of the observed value under the
  same distribution to the log-likelihood, log p(y | θ). The data's times
  replace the `emit_schedule` (§12.1.1).

Because only simulation uses the emission times, they are set by
`emit_schedule`, in one of two forms:

```
emit_schedule = every 7 'days                       # a recurring cadence
emit_schedule = at [7 'days, 14 'days, 28 'days]    # listed times, with the unit on each entry
```

A bare `every` or `at` field is refused with E272.

---

## 13. Interventions

An intervention changes the model's state at stated times: it moves a share of
one compartment into another, adds individuals, or sets a compartment to a
value. Interventions are inactive unless a run enables them (§13.4), so one
model holds the baseline and every policy compared against it; `events {}`
(§13.5) uses the same grammar for changes that always happen. The guide is
`camdl docs getting-started` ("Interventions: scheduled discrete events").

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

Each entry is `NAME : ACTION SCHEDULE`. The actions are specified in §13.1 and
§13.6, the schedules in §13.2 and §13.7, families of interventions in §13.3,
and the order in which effects due at one time apply in §13.10.

Related sections: §13.9 (interventions triggered by surveillance), §17
(scenarios, which enable interventions), §25.5 and §25.10 (how interventions
lower to the IR), §26 (the error table).

### 13.1 Actions

There are three actions. `transfer` and `add` are written as calls; a `set`
is written inside the block form as `COMP = EXPR`, and there is no `set(...)`
call.

```camdl
# transfer and add — function-call action forms
NAME : transfer(fraction = EXPR, from = COMP, to = COMP) at [...]   # move fraction
NAME : transfer(count = EXPR, from = COMP, to = COMP) at [...]       # move count
NAME : add(COMP, EXPR) at [...]                                       # add a count

# set — block form, one or more `COMP = EXPR` assignments plus a schedule
NAME : { COMP = EXPR  at = [...] }                                    # override value
```

**`transfer(fraction = f)`** moves `delta = floor(source · f)` individuals,
with `delta` computed from the state before the transfer and then subtracted
from the source and added to the destination together. A fraction outside
`[0, 1]` is clamped to that interval when the transfer fires. When the source
has an Erlang dwell (`via erlang`, §9.4.1), `source` is its total over its
stages: the floor is taken once and each stage gives up its share.

**`transfer(count = n)`** moves `min(round(n), source)` individuals. A count
that evaluates below zero when the transfer fires, whether a literal or a
parameter or expression that is negative at the parameters in use, stops the
run with a negative-count error naming the source compartment, on every
backend and under inference; the compiler does not check it. A count of zero
moves nobody. To move people the other way, write the transfer in the other
direction.

**`add(COMP, EXPR)`** and the block form are specified in §13.6.

**Targets of `set` and `add`.** Each names one compartment cell. On a
stratified compartment the bare name (`I`) is a family of cells, so the target
is written as the expanded cell name (`I_child_p1 = ...`, `add(I_child_p1, 5)`)
or indexed on every dimension (`I[child, p1] = ...`, `add(I[child, p1], 5)`).
Inside an indexed family (§13.3) the indices may be the family's binders,
including a `c in compartments` binder: `c[a] = ...`. A stratified family, an
unknown name, or an index that leaves a dimension out is **E265**; for a family
the hint lists its cells. On a compartment with an Erlang dwell, `add` enters
the first stage (`add(E[pre], 5)`) and `set` is **E366**; either verb takes one
stage written as `stage(E[pre], 2)` (§9.4.1).

**Pairing in a bare stratified `transfer`.** `transfer(from = S, to = V)` with
bare names on stratified compartments expands into one transfer per cell
(§25.10). The endpoints pair cell for cell, and only when they are declared
with the same dimensions in the same order. The compiler compares the declared
dimension lists, not the expanded cell names, so two dimensions that share level
names (`age = [low, high]` and `risk = [low, high]`) do not pair. Any other
combination (one endpoint stratified and the other not, different dimensions,
or the same dimensions in a different order) is **E237**, which names both
shapes. An endpoint indexed on every dimension, or written as one expanded cell
name (`S_pre`, or a stage cell `I_pre_s3`), is a single cell and carries no
dimensions, so `transfer(from = S[child], to = S[adult])`,
`transfer(from = S[child], to = V)` with `V` unstratified, and
`transfer(from = I_pre_s3, to = Q[pre])` each move one cell. The `fraction` is
one expression shared by every cell; coverage that varies by stratum is written
as an indexed family (§13.3).

**A bare endpoint inside an indexed family** is **E239**. In `vacc[a in age] :
transfer(from = S, to = V)` each instance would fan out over every cell, so
each cell would receive the transfer once per instance, and with `P` strata the
realised coverage would be `1 − (1 − f)^P` instead of `f`. Write
`from = S[a], to = V[a]`, or drop the binder to fan out once.

**`count` on a bare stratified transfer** is **E238**. A fraction is
scale-free and applies to every cell; a count is absolute, and applied per cell
it would move `count` individuals out of each stratum, multiplying the intended
total by the number of cells. Write `fraction =`, or index the transfer so each
instance names one cell. `count` is accepted wherever the transfer resolves to
a single cell.

### 13.2 Scheduling

A schedule is a list of times or a recurring cadence.

```
NAME : ACTION at [TIME, ...]     # times in model time_unit
```

```
NAME : ACTION {
  every = DURATION             recurring interval
  from  = DURATION             start of recurring (default: t_start)
  to    = DURATION             end of recurring (default: t_end)
}
```

A `transfer` or `add` takes either form. A `set` takes only a list, written
inside its block as `at = [...]`. An `at` list accepts instants and splices
`date_range(...)` (§2.1), and inside an indexed family it may read a table or
parameter indexed by the family's binders (§13.3). `every … at_day …` is a
third form, for `add` only (§13.7).

**Anchored mode.** Under `origin = date(...)`, `every` must be an Exact
duration (§2.1): a calendar cadence such as `every = 1 'months` is **E322**,
whose hint names `date_range(<start>, <end>, calendar_months = 1)` in an `at`
list for month-aligned firings and `every = 30 'days` for a fixed cadence.
`from`, `to` and the entries of an `at` list are instant positions, under the
one-step calendar rule of §2.1: `from = 6 'months` is
`add_calendar_months(origin, 6)`, and a form that rule refuses is **E321**.
Without an origin a month is a fixed span and `every = 1 'months` is accepted.

**When a scheduled time fires.** Every schedule resolves to a list of declared
times, and each backend fires them on its own clock, so a simulation and a fit
on the same backend apply the same effects.

- An effect fires only inside the run's window, `t_start ≤ t ≤ t_end`, to
  within `1e-9 · dt` on a discrete-time backend and `1e-10` on a continuous
  one. A declared time outside the window does not fire.
- **Discrete-time backend** (chain-binomial, in simulation and under every
  inference method that steps it). An effect fires at the end of a step of the
  run's grid `t_start + k·dt`, with `k = ⌊(t − t_start)/dt + ½⌋`: the nearest
  step, a tie going to the later one. `at [2.4]` at `dt = 1` from `t_start = 0`
  fires at 2; from `t_start = 0.5`, an effect at 1 fires at 1.5. When `t_end` is
  off the grid the last step is shortened to end at `t_end`, and an effect
  inside that step fires at the nearer of its two ends, a tie going to `t_end`,
  with a warning naming the effect, its time and the end it fires at. One
  intervention fires at most once per step: `at [2.3, 2.4]` at `dt = 1` fires
  once, at 2. A recurring schedule whose period is shorter than `dt` would put
  two fires on one step; the run is refused at its start (and a fit at its
  pre-flight) with an error naming the two times, and the fix is a finer `dt`
  or a longer period.
- **Continuous-time backends** (ODE, Gillespie) fire an effect at its declared
  time, so `at [2.3, 2.4]` fires twice.
- An effect at `t_start` (`k = 0` on the discrete-time backend; within `1e-10`
  of `t_start` on a continuous one) applies to the starting state, before the
  `t_start` row is recorded or an observation there is scored (§13.10). A
  recurring schedule's default `from` is `t_start`, so its first fire is such an
  effect. A run resumed from a saved state (a contrast fork, a forecast from
  `--init-state`) does not apply it again, because the saved state already
  includes it.

### 13.3 Indexed Interventions

An index binder declares a family of interventions, one per level:

```camdl-doctest-preamble iv-patch
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
```

```camdl preamble=iv-patch
interventions {
  # Declares sia_north, sia_south, sia_east (one per patch)
  sia[p in patch] : transfer(fraction = vacc_eff * sia_cov, from = S[p], to = V[p]) at [180, 545]
}
```

The header is `NAME[VAR in DIMENSION, ...]`, followed by the action and either
schedule form. A trailing `where` guard restricts the levels, as on a
transition (§9.3): `x[a in age] : add(I[a], 5) at [10] where a != child`.

The members share a base name, the unindexed name (`sia` above). In a
scenario's `enable` or `disable` list and on the command line, the base name
selects every member; a member is selected alone by its expanded name
(`sia_north`). See §17.

**Members must differ.** Every member fires, so two members with the same
actions on the same schedule apply the action twice:
`sia[r in round] : transfer(count = n, from = S, to = V) at [10]`, whose body
never mentions `r`, moves `2n` individuals where it reads as `n`. A family
whose expansion yields two such members is **E421**, which names the binder
that does not distinguish them. Use the binder in the action (`from = S[r]`) or
in the schedule (an `at` entry indexed by `r`), or drop it. The check compares
the expanded members: a binder used only in the schedule gives distinct fire
times and is accepted, and so is a `where` guard that leaves one member. It
applies to interventions, events (§13.5) and reactive interventions (§13.9),
with every action.

**Indexed `set` and `add` targets.** Inside a family the target of a `set` or
an `add` may be indexed by the family's binders. With a `c in compartments`
binder, one family covers every compartment:

```camdl preamble=iv-patch
events {
  # every compartment in each patch halved at t = 200 (a cull, or a census correction)
  halve[c in compartments, p in patch] : { c[p] = 0.5 * c[p]  at = [200] }
}

interventions {
  reseed[p in patch] : add(I[p], 5) at [300]
}
```

`halve` expands to one `set` per cell (`S_north = 0.5 * S_north`, and so on for
each compartment and patch). Every instance due at one instant reads the same
state from before the effects (§13.10), so a family whose right-hand side reads
other cells, such as an age shift `c[a] = … c[b] …`, is a simultaneous map
whose result does not depend on the order of declaration or expansion. The
target indexes every dimension of its compartment. A `set` does not fan out
over an omitted dimension as a transition does (§9.5), because `c[a]` on its
right-hand side would be ambiguous between the cell and the stratum total:
`S[a]` on a compartment stratified by `[age, vx]` is **E265**, naming the
unindexed dimension. Named indices (`S[vx = v, age = a]`) are placed in
declaration order, as everywhere else.

**Per-level times from a table.** The binder is in scope in the schedule, so a
table holds a schedule that differs by level:

```camdl
tables {
  sia_day : patch × round = read("data/sia_schedule.tsv")
}

interventions {
  sia[p in patch] : transfer(fraction = vacc_eff * sia_cov, from = S[p], to = V[p])
    at [sia_day[p, 0], sia_day[p, 1]]
}
```

`sia_day[p, 0]` is looked up at compile time, so each member has its own
constant fire times.

### 13.4 Activation

An intervention is off unless the run enables it, with a scenario's `enable`
list (§17) or on the command line:

```bash
camdl simulate model.camdl --enable sia_round_1 --seed 42
```

`--enable` and `--disable` may be repeated and cannot be combined with
`--scenario` (§21).

---

### 13.5 Events

An event is a scheduled change that every run applies. Events take the action
grammar and schedules of interventions, indexed families, `where` guards and
every action included, but they are not toggled: a scenario's `enable` or
`disable` list naming an event is **E267**, an unknown intervention. They hold
structural processes (cohort entry, seasonal migration, importation seeding);
policies compared across runs (vaccination campaigns, school closures) are
interventions.

```camdl-doctest-preamble events-demo
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
```

```camdl preamble=events-demo
events {
  cohort_entry : add(S, cohort * birthrate(t) * pop(t) * 365.25 'days)
    every 365.25 'days at_day 251

  importation : add(I, 10) at [30]
}
```

### 13.6 The `add` Action

```
add(COMPARTMENT, EXPR)
```

Adds `round(EXPR)` individuals to an integer compartment, or `EXPR` to a real
one. The amount cannot be negative: an `add` whose amount evaluates below zero
when it fires stops the run with a negative-count error naming the compartment,
on every backend and under inference. The compiler does not check it, so
`add(I, -5)` compiles and fails when it fires. A removal is a `transfer` to
another compartment.

### 13.7 The `at_day` Schedule

`every PERIOD at_day DAY` fires an `add` on one day of each period:

```camdl
NAME : add(COMP, EXPR) every PERIOD at_day DAY
```

The form is accepted only on `add`; on `transfer` or a `set` it is a syntax
error (**E001**), and those actions use the forms of §13.2. `at_day` is the
phase within the period measured from `t = 0`. The fire times are
`at_day + k · period` for `k = 0, 1, …`, starting at the smallest `k` whose
time is at least `t_start`, and each fires as §13.2 states. `every 365.25 'days
at_day 251` fires on day 251 of each year; a run starting at `t = 100` first
fires at `t = 251`. The schedule fires once per period, which a hand-written
`mod(t, period)` condition does not guarantee when `dt` does not divide the
period; a `dt` coarser than the period is refused as §13.2 states.

Under `origin = date(...)` the offset must be an Exact duration (§2.1). It is
folded to one number, so `at_day 1 'months` would mean the average month
(30.436875 days) in every period; it is **E322**. The offset is written in
`'days` or `'weeks`, and calendar-aligned firings are listed with
`at [date(...), ...]`.

---

### 13.8 Balance Constraint

A `balance {}` block keeps one compartment equal to an expression of the
others, typically a known population size minus the other compartments. After
transitions, events and interventions have applied in a step, the target is
overwritten with the expression (§13.10).

```camdl-doctest-preamble balance-demo
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
```

```camdl preamble=balance-demo
balance {
  R = pop(t) - S - E - I
}
```

This is pomp's `R = nearbyint(pop) - S - E - I` pattern, for a model whose
population trajectory is given externally and whose birth and death rates do
not reproduce it exactly. The target is exempt from the non-negativity check: a
negative value means the expression and the dynamics disagree, and is reported
separately. An event that adds people without a source (`add(S, 20000)`)
raises the other compartments, and the target decreases to keep the total.

Balance is applied on the chain-binomial backend only. A model with a
`balance {}` block is refused on the ODE and Gillespie backends with a
capability error naming the block. A drawn initial condition on the target is
**E345** (§15.4).

### 13.9 Reactive Interventions

A reactive intervention fires when surveillance data cross a threshold, instead
of at a time fixed in advance, as in an outbreak response that runs a
vaccination campaign after detected cases exceed a count. It is a third kind of
effect beside interventions and events. The guide is `camdl docs features`
("Reactive interventions").

```camdl-doctest-preamble reactive-demo
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
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_afp    ~ poisson(rate = rho * projected)
  }
}
```

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

The `when` predicate is a boolean over **trigger inputs**, which read the
observations a stream emits and never the latent state:

- `observed(stream)`, the most recent observed value of an observation stream;
- `sum_observed(stream, window = D)`, the sum of the stream's observations in
  the trailing window `D`.

These are valid **only** inside a `when` predicate; using `observed(...)` in a
transition rate or any other model expression is an error (it would read data
the rate has no business reading). The predicate combines comparisons with
`and` / `or` / `not`; each comparison has exactly one trigger input on one side
and a constant or parameter threshold on the other. A threshold computed from
a parameter, such as `>= 2 * thr`, is refused with E378; declare the threshold
itself as the parameter.

A trigger input outside a predicate is **E278**. A comparison without exactly
one trigger input, or a predicate that is not a comparison or a combination of
comparisons, is **E273**, and a trigger naming an undeclared stream is
**E279**.

The policy body takes four keys; any other is **E106**.

| Field      | Meaning                                                            |
| ---------- | ----------------------------------------------------------------- |
| `action`   | The state change to apply — same `transfer` / `add` grammar as interventions. |
| `after`    | Non-negative lag between the trigger firing and the effect (default `0`). |
| `once`     | `true` (default) fires at most once; `false` allows repeats.      |
| `cooldown` | Minimum time between firings when `once = false`. Mutually exclusive with `once = true`. |

A negative `after` is **E274**. `cooldown` with `once = true`, or with `once`
omitted, is **E276**. The action is a `transfer` or an `add`; a `set` block is a
syntax error (**E001**).

**Meaning.** The predicate is a level test, evaluated at each time a stream
emits an observation and not between them, and gated by `once` and `cooldown`.
`sum_observed(stream, window = D)` sums the emissions in the half-open window
`(now − D, now]`, so an emission at `now` is included. A trigger input over a
stream that has not yet emitted reads 0. When a policy fires, its effect is
scheduled at `trigger_time + after`, and `cooldown` is measured from the
trigger time, not from the delayed effect.

`window`, `after` and `cooldown` are each folded to one fixed span at compile
time. Under `origin = date(...)` a calendar duration in any of them
(`after = 1 'months`) is **E321**: a calendar month is 28 to 31 days, so the
span is stated in `'days` or `'weeks`.

The trigger reads the observation drawn at each emission, the value a
surveillance system would report, and never the latent state.

**Activation.** A reactive intervention is a policy, toggled per run like an
intervention, so a run without it and a run with it compare the world without
a response and the world with one. It is inactive by default and enabled with
`--enable`; a policy that is not enabled is dropped from the run.

A scenario cannot enable or disable a reactive policy: naming one in a
scenario's `enable` or `disable` list is **E267** (unknown intervention), as
only `interventions {}` names are in scope there. The reason is the trigger's
input: the emission times its stream is read at are fixed when the model is
compiled, ending at the model's `simulate { to }` horizon, so a scenario that
moved the horizon would run the dynamics past the last time the policy can
observe.

**Support.** An enabled reactive policy runs in forward simulation on the
chain-binomial backend, and each firing is recorded in the run's
`reactive_log.tsv` artifact (`camdl simulate --reactive-log` also writes a
copy). On the ODE and Gillespie backends and under inference, a run with an
enabled reactive policy is refused with a `REACTIVE_INTERVENTIONS` capability
error. A policy that is not enabled is dropped, so a run without `--enable` is
accepted on every backend.

### 13.10 Within-substep effect ordering

When several effects are due at one time they apply in a fixed order shared by
the backends, which determines the state an observation or a later effect
reads. On the discrete-time (chain-binomial) backend, one step runs:

1. **Inflow events** (`add`, and source-less `--> D` events), computed from
   the state at the start of the step and combined with the transition draw.
2. **Transitions**, which advance the compartments.
3. **Draining events** (`transfer`) and `set`, applied to the state after the
   transitions.
4. **Scheduled interventions**, in declaration order, on the state after
   stage 3.
5. **Balance**, which overwrites its target (§13.8).
6. **The non-negativity check**, from which the balance target is exempt.
7. **The trajectory row** for the step, recorded after all of the above.

The continuous backends (ODE and Gillespie) have no transition stage at an
effect time and apply the same order there: inflow events are computed from
the state just before the effect time, draining `transfer` events and `set`
apply to the state after the inflow, and scheduled interventions follow in
declaration order. An `add(X, 10)` event and an `{ X = 5 ... }` event due at
the same instant therefore leave `X = 5` on every backend, whichever is
declared first, and a draining `transfer` due beside an inflow into its source
moves a share of the count after the inflow.

**Draining events and `set`.** Within stage 3 every action resolves against
the same state (after the inflow, and on the discrete-time backend after the
transitions), and their changes add: two draining `transfer`s from `X = 10`,
of fractions 0.5 and 0.2, move 5 and 2 and leave `X = 3`, in either order. A
`set` cannot share its compartment with another action of that stage. A second
`set` on it, or a `transfer` from or into it, at the same instant is an error
on every backend, since the changes would add to a value neither event names:
from `X = 10`, `{ X = 5 ... }` beside a half drain of `X` would give `X = 0`.
The two are given different times or merged into one. An inflow `add` beside a
`set` is accepted: it acts in stage 1, and the `set` then overwrites the
result.

**Drains that could overdraw.** Each draining `transfer` is capped only by the
shared state, so two that together take more than the compartment holds leave
it negative, and the model does not say which one goes short: two
`transfer(count = 6, from = X, ...)` from `X = 10` would leave `X = -2`. Among
the events due at one instant it is therefore an error, on every backend, to
drain one compartment with two or more `transfer`s when at least one moves a
`count`, whatever the compartment holds, so the error is raised the first time
the events are due together, identically on every run and every particle. It
is also an error when the `fraction`s of the drains sum to more than 1. A
fraction is evaluated at the instant, as the transfer evaluates it, so a sum
above 1 that comes from the parameters is refused for every particle.
Fractions summing to at most 1 (0.3 and 0.2) never overdraw and are accepted.
The drains are given different times or merged into one transfer. A single
`transfer(count = n, ...)` moves at most what its source holds: from `X = 10`,
`count = 50` moves 10. A stage that leaves an integer compartment negative (a
`set` to a negative value) is an error on every backend.

**Compartments with an Erlang dwell.** A `transfer` out of a compartment with
an Erlang dwell (§9.4.1) takes part in these rules as a drain of each of its
stages, with its one `fraction` (or `count`) on each. Two transfers out of the
same staged compartment at the same instant are an error on every backend,
whatever their fractions and whatever the compartment holds: each divides its
own total among the stages, and both can assign the individual left over by
the floor to the same stage (from stages holding (1, 1), two transfers of 0.5
would each take the person in stage 1). The error names both events and the
stage they share; the transfers are given different times or merged into one.
A transfer out of a staged compartment beside a `transfer` that names one of
its stages (`from = stage(E[pre], 2)`) is checked by the sum of their
fractions, like any two fraction transfers, and cannot overdraw when the sum is
at most 1: from a stage holding `n`, the staged transfer takes at most
`ceil(n · f)` and the other `floor(n · f')`, which together never exceed `n`.
Scheduled interventions are not subject to these rules, because they apply one
after another (stage 4).

**At `t_start`.** The effects due at the start of the run (§13.2) apply to the
`init {}` state in the same order, without a transition stage, on every
backend: inflow events from the `init {}` state, then draining events and `set`
on the state after the inflow, then interventions in declaration order, then
`balance`, then the non-negativity check. The `t_start` row is recorded, and an
observation at `t_start` scored, after all of them. With `init { I = 0 }`,
`seed : add(I, 10) at [0]` and
`cull : transfer(fraction = 0.5, from = I, to = R) at [0]`, the `t_start` row
is `I = 5`, `R = 5` on every backend. When nothing is due at `t_start` the
`init {}` state is recorded as it is, and a `balance {}` target is not
rewritten. An action's amount is evaluated at `t = t_start`.

Recorded output is therefore the state after the effects. An observation
emitted at a time when an intervention fires, such as a vaccination
`transfer`, reads the state after the vaccination, at `t_start` as at any other
time.

---

## 14. Timepoints and Reserved Identifiers

A `timepoints {}` block is accepted and checked, but the names it declares are
not in scope anywhere in the model. Each entry is an instant position (§2.1)
and is checked as one: under an origin, `peak = origin + 6 'months` is the
calendar date six months after the origin, by the one-step calendar rule, and
`peak = origin + 6 'months + 6 'months` is **E321**. A reference to an entry's
name in an expression or a schedule is **E100**, an undeclared name.

```camdl
timepoints {
  midpoint     = 1 'year
  intervention = 180 'days
}
```

Related sections: §2.1 (instant positions), §18 (the `simulate` block).

### 14.1 Built-in Timepoints

`t_start` and `t_end` are the run's start and end times, set by
`simulate { from, to }` (§18). They are reserved names (§14.2), and they are
the defaults of a recurring schedule's `from` and `to` (§13.2). They are not in
scope in an expression: `at [t_start]`, `from = t_start` and a rate that reads
`t_end` are each **E100**, an undeclared name. A model without a `simulate {}`
block (as under `camdl check`) is compiled with `t_start = 0` and `t_end = 100`,
without a warning.

### 14.2 Reserved Identifiers

A name can be unavailable for a declaration in three ways, each with its own
diagnostic.

**Reserved names.** Declaring a compartment, parameter or `let` binding with
one of these names is **E100** ("name '…' is reserved …"):

```
t          # current simulation time
t_start    # simulation start time (from simulate block)
t_end      # simulation end time (from simulate block)
dt         # current substep length (used inside rate expressions)
pi         # the constant π
e          # Euler's number
```

```
ERROR E100: parameter name 't_end' is reserved for simulation time
```

**Keywords.** The lexer reads these as keywords, so one used as a name is a
syntax error, **E001**. They include the block keywords (`compartments`,
`parameters`, `tables`, `forcing`, `transitions`, `observations`,
`interventions`, `events`, `reactive_interventions`, `ode`, `output`,
`simulate`, `init`, `scenarios`, …), the type keywords (`rate`, `probability`,
`positive`, `count`, `real`, `integer`, `instant`, `duration`), and the
operator, iteration and field keywords (`sum`, `consecutive`, `where`, `let`,
`if`/`then`/`else`, `and`/`or`/`not`, `in`, `by`, `from`, `to`, `every`,
`until`, `at`, `when`, `action`, `origin`, `columns`, `emit_schedule`, …).

**Function and distribution names** are neither reserved nor keywords. They are
read as functions only in call position, so a parameter may take one of these
names, at a cost to readability. They include the calendar functions
(`add_calendar_months`, `add_calendar_years`, `date`, `date_range`; there is no
`add_calendar_days` or `add_calendar_weeks`), the rate wrappers
(`overdispersed`, `deterministic`), the likelihood families (`poisson`,
`neg_binomial`, `normal`, `binomial`, `beta_binomial`, `beta`, `bernoulli`,
`diagnostic_test`), the projection name `projected`, and the scenario names
`baseline` and `scenario`.

---

## 15. Initial Conditions

The `init {}` block gives each compartment's value at the start of the run,
either as an expression (`=`) or as a draw from a distribution (`~`, §15.4).
The guide is `camdl docs getting-started` ("Init: starting conditions").

### 15.1 Un-Stratified Models

```camdl-doctest-preamble init-sir
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
```

```camdl preamble=init-sir
init {
  S = N0 - I0
  I = I0
}
```

A compartment without an entry starts at 0. An entry may read parameters,
tables (§15.3) and other entries (§15.4). Entries are evaluated at the start of
the run, `t = t_start` (`simulate { from }`), so an entry that reads `t` or a
forcing takes its value at that instant. An entry has the dimension of a count;
`S = t` is **E313**.

An entry for a compartment with an Erlang dwell (`via erlang`, §9.4.1) places
its value in the first stage, and `stage(E, 2) = 10` places a value in another.
No form spreads one value over the stages.

### 15.2 Stratified Models

On a stratified compartment, a bare name in `init {}` is **E277**: the
compiler does not choose how to divide a total among the strata, so each cell
is named.

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

An entry's target and its right-hand side may use indexed names, and an indexed
parameter may be written indexed or as its expanded name:

```camdl-doctest-preamble init-region
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
```

```camdl preamble=init-region
init {
  S[urban] = N0[urban] - I0   # indexed syntax — preferred
  S[rural] = N0_rural         # mangled form — still accepted
}
```

Named indices are accepted, in any order:

```camdl-doctest-preamble init-strat
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
```

```camdl preamble=init-strat
init {
  S[age = child, patch = p1] = 100000
}
```

A cell without an entry starts at 0, so a model seeded in one patch lists only
that patch's cells.

### 15.3 Init from Tables

An index binder on an entry declares one entry per level, and a table (§6)
supplies per-level values:

```camdl-doctest-preamble init-table
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
```

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

`N0[p]` is looked up when the model is compiled, so each expanded entry holds
its own constant. A parameter reference (`I0`) stays an expression evaluated
when the run starts. There is no allocation function that divides a total
among strata.

### 15.4 Drawn Initial Conditions

An entry written with `~` starts the compartment at a draw from a
distribution, with the reading `~` has for a parameter prior (§4) and an
observation (§12). The guide is `camdl docs features` ("Uncertain initial
conditions").

```camdl-doctest-preamble init-drawn
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
```

```camdl preamble=init-drawn
init {
  I ~ poisson(rate = I0)   # the number introduced is a DRAW around I0
  S = N0 - I               # reads the drawn I, so S + I = N0 exactly
}
```

Entries are evaluated in dependency order, so `S` reads the value `I` was
drawn as, and `S + I = N0` holds on every draw without a `balance {}` block.

The admissible laws depend on the compartment's kind:

| compartment | admissible laws                                                                  |
| ----------- | -------------------------------------------------------------------------------- |
| integer     | `poisson(rate = ..)`, `binomial(n = .., p = ..)`, `neg_binomial(mean = .., r = ..)` |
| `: real`    | `normal(mean = .., sd = ..)`                                                      |

A law of the wrong kind for its compartment is **E344**. A family that is not
an initial-state law is **E343**: `bernoulli`, `beta` and `beta_binomial`
describe a measurement of a compartment, and belong in `observations {}`.
`binomial`'s `p` given as a parameter declared with a kind other than
`probability` (a `count`, say) is **E344**, naming the parameter. A law on the
`balance {}` target is **E345**, because the balance stage overwrites the
target after every substep, so the draw would be discarded before the first
step.

`neg_binomial` describes introductions that are clustered: its dispersion `r`
sets how much more variable the count is than a Poisson count of the same mean,
and a large `r` approaches the Poisson.

**What the data can say about the parameter.** On the stochastic backends an
integer compartment written with `=` is rounded, so `I = I0` starts at
`round(I0)` and the likelihood is a step function of `I0`, flat within each
rounding interval; a fit then learns less about the seed than the data contain.
The ODE backend starts from `I0` unrounded. A drawn initial condition gives a
likelihood that is smooth in its parameter. `camdl docs inference`
("Initial-state parameters") describes what this means for a posterior.

**Under inference.**

- **`pgas`.** `log p(x₀ | θ)` is a term of the complete-data likelihood and is
  written to the `initial_state_ll` column of each chain's trace, so the law's
  parameters are estimated from the data.
- **`if2`, `pfilter` and `pmmh`.** Each particle draws its own initial state
  from its own random stream, so the particle filter integrates over
  `p(x₀ | θ)` instead of conditioning on one draw. That spread is what makes
  `ic_free = true` meaningful under `pfilter` and `pmmh`; with a deterministic
  `init {}` every particle starts at the same state, the first reweighting
  cannot distinguish them, and `ic_free` is refused.
- **Fits on the ODE backend** start every compartment at its law's mean.

**A negative starting count.** An integer compartment cannot start below zero,
and what happens when one would depends on whether its entry reads a draw.

- **The entry is a function of the parameters alone**, such as
  `R = N0 - S0 - I0` with `I0` large enough. Every particle starts at the same
  negative value, so the parameters are infeasible: under `pfilter`, `pmmh` and
  `if2` they score a log-likelihood of −∞ and the fit rejects them and
  continues, as for a value outside a bound. `simulate` stops with an error
  (`NegativeCount { cause: InitialState }`) naming the compartment and its
  value.
- **The entry is drawn, or reads a drawn entry**, such as `S = N0 - I` above
  when `I` is drawn larger than `N0`. Some particles would start negative and
  others not, and discarding only the negative ones would condition the fit on
  the draw having been feasible, which the model does not state. This is an
  error on every command, `fit` and `pfilter` included
  (`NegativeCount { cause: InitialStateDrawn }`), raised when the state is
  drawn. The fix is an `init {}` that cannot go negative, such as
  `I ~ binomial(n = N0, p = ..)`, which never exceeds `N0`, in place of a
  `poisson` draw subtracted from `N0`.

An entry depends on a draw when it is written with `~` or reads, directly or
through other entries or `let` bindings, a compartment that is.

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

**In anchored mode**, `every` must be classified `Exact`, as for an
intervention's `every` (§13.2): a regular schedule has one fixed step, so
`every = 1 'months` would place snapshots 30.436875 days apart and drift off the
month boundaries. It is **E322**; list calendar-aligned times as
`at = [date_range(origin, add_calendar_years(origin, 1), calendar_months = 1)]`,
or write `every = 30 'days`. The `at` entries are instant positions (§2.1): a
single calendar step in one is its calendar date (the one-step calendar rule),
and any other calendar duration is **E321**.

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
The final trajectory row is therefore the last output time at or before `t_end`;
a firing after that time (for example between a final weekly output at 14 and a
`t_end` of 20) appears in no row.

Synthetic observations (forward simulation) are not part of the trajectory
output; they are produced by the simulate command's `--obs` family of flags
(§21), generated by the runtime's `sample_observations` method from the
observation model definitions.

### 16.3 Per-run overrides (CLI and config)

The cadence and the columns can also be set per run, without editing the model
— useful for exploratory runs and for large stratified/spatial models whose
`flow_*` columns dominate the output. These are `simulate` flags. (Only
`simulate` writes trajectories, so the view applies there; `fit.toml` has no
trajectory output.)

| Flag                | Effect                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `--output-every N`  | One row every `N` time-units, overriding the model's `output { every }`. A plain number in the model's `time_unit` (like `--dt`). |
| `--no-flows`        | Drop every `flow_*` column.                                                                                                       |
| `--columns A,B,…`   | Restrict to this allow-list of output column names (compartments and/or `flow_<name>`); emitted order follows the model.          |

```bash
# weekly latent state, no flow columns — on a 23-patch spatial run the
# patch×patch flow columns are ~93% of the output
camdl simulate polio.camdl --draws prior -n 100 --output-every 7 --no-flows
```

These overrides participate in run identity. `--output-every` rewrites the
model schedule (it rides the model digest, re-keying only runs that use it);
`--no-flows` / `--columns` ride the `config` level, because a column subset is a
distinct, reproducible artifact — a content-addressed leaf cannot share a
`run_id` with the full one. (Introducing this view bumps the `config` schema
version, so all sim `run_id`s shift once — a deliberate, versioned turnover;
existing cached sims re-run on next use.) An unknown `--columns` name is a hard
error that lists the valid columns.

### 16.4 Derived quantities (`quantities {}`)

`quantities {}` is a top-level block — a sibling of `observations {}` and
`output {}`, not nested in either — that declares **derived quantities**:
summaries computed from a run and reported alongside the trajectory, but never
scored as data. It is the non-scored twin of an observation. Where
`observations {}` defines what the likelihood _sees_, `quantities {}` defines
what you want _read back_ — a peak size, an attack rate, the time an outbreak
takes off — so a summary no longer has to be smuggled through a fake scored
stream.

```camdl-doctest-preamble quantities-demo
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
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases ~ neg_binomial(mean = rho * projected, r = k)
  }
}
```

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
are never interpolated). `TIME` is an instant position (§2.1): a constant time
expression — `date("2026-08-10")` under a declared `origin` (where a bare number
draws **W325**, as an `at [...]` entry does, and a `'months`/`'years` literal
follows the one-step calendar rule: `value_at(I, 6 'months)` reads `I` six
calendar months after the origin), or a number in model time —
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
`'months`/`'years` offset is rejected too (**E321**): the date `last_obs` stands
for is known only when the model runs, and a calendar month from a date depends
on which date it is, so the resolved time would move with the data. Note that camdl writes durations with a leading tick
(`1 'weeks`); a bare word (`1 weeks`) is **E115**, which names the tick form.
The command line takes the opposite convention (`--to "last_obs + 8 weeks"`,
bare words), because a tick is a shell-quoting hazard; under a declared
`origin` it refuses a month or year offset the same way
(`--to "last_obs + 6 months"` asks for `"last_obs + 183 days"`).

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
`model_identity` unchanged and orphans no completed fit. This holds for the
file-header block that documents the whole model (§1) exactly as it does for a
parameter's, which is why that block, and not `description = "…"`, is where a
model's real description belongs: a description nobody dares correct is worse
than none. Both rules exist so
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

### 16.5 Counterfactual contrasts (`contrasts {}`)

A `quantities {}` recipe reduces **one** run. A `contrasts {}` entry compares
**two**: it is the block that answers "how many deaths did the campaign avert?"
Each entry is arithmetic over *run-rooted* operands of the form
`<run>.<namespace>.<member>`, where `<run>` is a preset declared in
`scenarios {}` (§17) or the reserved name `fitted` (the fitted model with no
scenario overlay), and `<namespace>` is `quantities` or `observations`.

```camdl-doctest-preamble contrast-sir
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
interventions {
  sia_campaign : transfer(fraction = 0.3, from = S, to = R)
    at [60 'days]
}
init { S = N0 - I0  I = I0 }
simulate { from = 0 'days  to = 180 'days }
scenarios {
  no_sia   { set = { beta = 0.4  gamma = 0.15  N0 = 10000  I0 = 10 }
             disable = [sia_campaign] }
  with_sia { set = { beta = 0.4  gamma = 0.15  N0 = 10000  I0 = 10 }
             enable  = [sia_campaign] }
}
```

```camdl preamble=contrast-sir
quantities {
  total_inf = final(R)
}

contrasts {
  averted      = no_sia.quantities.total_inf - with_sia.quantities.total_inf
  averted_frac = (no_sia.quantities.total_inf - with_sia.quantities.total_inf)
                 / no_sia.quantities.total_inf
}
```

**Why this is a language construct rather than two runs and a subtraction.**
Differencing two independently-simulated arms by hand mixes the intervention
effect with the Monte Carlo noise of two unrelated random draws. The reducer
behind `contrasts {}` instead *forks* both arms from a shared trajectory: it
diffs the two runs' live intervention sets, finds the toggled intervention, and
branches at the last saved snapshot strictly before the instant that
intervention fires on the arms' chain-binomial grid (§13.2: declared at 2.4 with
`dt = 1`, it fires at 2, so with snapshots on the integer grid the fork is at
1), so both arms share one history up to the intervention and one per-draw seed. The
difference is then attributable to the intervention rather than to sampling.
Results are banded over the forkable draws into `contrasts/<name>.tsv`. A
contrast whose two arms differ only in a parameter — no intervention is toggled
— has no fork point, and is skipped with a note rather than silently reported.

**Division by zero is censoring, not an error.** A stochastic arm can
legitimately end a draw with a zero count, so a ratio such as
`(baseline.x - vaccinate.x) / baseline.x` has a zero denominator in some
posterior draws. A finite numerator over a zero denominator (`/`, and likewise
`%`) is *undefined for that draw*: the draw is censored, by the same mechanism
as a `time_of_max` that never fired. For a scalar contrast the draw drops out of
the band; for a series contrast only the affected time cells drop out, so the
other snapshots of that draw still count. The band's `n_used` column reports,
per cell, how many draws it was computed over. Censoring propagates through
further arithmetic (a result touching a censored operand is censored). For each
contrast with at least one such draw, `fit predict` prints one line on stderr:

```text
fit predict: contrast 'rel': 3 of 200 draws undefined (denominator 0); excluded from the band, see n_used
```

The count is of draws with at least one undefined cell. If every draw is
undefined the band cells are empty and `n_used` is 0. This applies only to a
zero denominator: any other non-finite per-draw value (an overflow, or a
non-finite operand) is still a hard error, because it indicates a defect rather
than a property of the draw.

**Diagnostics.**

- **E292** — a run-rooted reference used outside `contrasts {}`. It is a
  contrast operand and has no value in a rate, `let` binding, or other
  per-instant expression.
- **E293** — a run-rooted reference inside a `quantities {}` recipe. The run is
  implicit there; drop the `<run>.` prefix.
- **E294** — an operand names a run that is not a declared scenario nor
  `fitted`, or a quantity/stream the model does not declare.
- **E295** — the body is not run-rooted arithmetic (a bare constant, a
  comparison, or an inline reducer appeared where `<run>.<ns>.<member>` operands
  combined with `+ - * /` are required).
- **E297** — the two arms have incompatible dimensions (`deaths - rate`).
- **E298** — two entries share a name; each lowers to one
  `contrasts/<name>.tsv`, so a duplicate would clobber its sibling.

Like `quantities {}`, a `contrasts {}` block is a derived report rather than an
input to the dynamics, so adding or changing one never re-keys a model's
`run_id`.

---

## 17. Scenarios

Patch-based modifications to the baseline. Baseline is the identity patch — the
model as defined, no modifications.

```camdl-doctest-preamble scenario-sia
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
```

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

`enable`, `disable` and `compose` take a bracketed list of names even for one
name: `enable = sia` is an error (**E123**, with a fix-it to `enable = [sia]`),
as is a list member that is not a name. `set` and `scale` are the only fields
that take a `{ ... }` block; `beta = { ... }` is an error (**E124**) whose hint
shows `set = { beta = ... }`.

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

```camdl-doctest-preamble scenario-vacc
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
```

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

::: {.callout-warning}
**`enable`/`disable`/`compose` append parent's list to
the child's.** A child writing `enable = [masking]` under a parent with
`enable = [vaccination]` gets `[vaccination, masking]`, **not** just
`[masking]`. To remove a parent's intervention in a child, use `disable`. The
compiler emits **W310** whenever this merge actually changes the child's
declared list, so the surprise is observable rather than silent.
:::

**Diagnostics:**

- **E353** — cycle in `extends` chain (includes the full chain in the message).
- **E354** — unknown parent scenario (suggests the closest name by edit
  distance).
- **E355** — chain depth > 5; treat as a code smell and factor common ancestors,
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

```camdl-doctest-preamble sir-basic
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
```

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
silently dropped; the accepted keys are `from`, `to`, `dt`, `integrator`. `to`
is required: a `simulate {}` block without it is `E106`, because the end of the
run has no default. `from` defaults to `0`. Both may be observation anchors, and
a run may override `from` (below).

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

### An observation-anchored start, and a start chosen per run

`from` may be an observation anchor too, under the same rules as `to`: a model
whose run should begin one week before its data begin says so, rather than
carrying a day that has to change with the data window:

```camdl
simulate { from = first_obs - 7 'days  to = last_obs }
```

`first_obs` is the start of the earliest observed period over the run's bound
streams, so for a stream with `covers = closing_at(time, 7 'days)` whose first
row is labelled day 5, `first_obs` is day −2, the day that week opened. A model
that should start where its first observed period opens writes
`from = first_obs`.

The start is resolved once per run, from the observation times the run binds,
by every command that resolves an anchored `to`, and a command with no data
refuses an anchored start by name. The resolved start is substituted into the
model the run is content-addressed by, together with every start the compiler
derived from it: the regular output schedule, a regular `emit_schedule`, and a
recurring intervention or event that declares no `from` of its own. A resolved
model is therefore the model that would have compiled with that start written
as a literal: two data windows are two runs, and one window reached by two
spellings is one run. `init {}` is evaluated at the resolved start (§15.1).

A run may also be given its start from outside the model file, so that one
model serves several data windows. Highest precedence first:

1. `camdl simulate --from SPEC` on the command line;
2. `[simulate] from = SPEC` in the fit config (`camdl docs fit-toml`), read by
   every command that reads the config (`fit run`, `pfilter --fit`,
   `profile --fit`, `survey --fit`, `simulate --fit`);
3. the model's own `simulate { from }`.

SPEC is the CLI time grammar of `--to`: a model-time number, a date under a
model `origin`, or an anchor with a bare-word offset (`"first_obs - 1 week"`).
The override is applied exactly as an anchored start is, so it enters the run's
identity through the model and moves the same derived starts. A stored fit
records the start it was fitted from, and `fit predict` and
`simulate --fit <run dir>` replay that start.

One case is refused rather than guessed. When the model's `from` is a literal
and a recurring intervention or event starts at that same time, camdl cannot
tell whether the schedule declared no `from` (and should move with the run's
start) or declared one equal to it (and should stay), so a different start from
the command line or the config is refused, naming the schedules. Writing the
model's `from` as an observation anchor removes the ambiguity, because a
schedule that declares no `from` then carries no number at all until the run
resolves it.

---

## 19. Content-Addressable Output

Outputs are stored in a **content-addressed store** under an `output_dir` you
choose, partitioned by artifact kind (`sims/` for `simulate` trajectories,
`fits/` for inference runs, and so on):

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

A model declares its parameters, their bounds and their priors (§4); their
values are supplied from outside the model file. This section specifies the
params file that carries a point of values, the prior syntax on a declaration,
and how both relate to the parameter grammar. Which source wins when several
set one parameter is the precedence chain of the run spec (§1.3).

### 20.1 Values

A params file is a flat TOML file with one numeric value per parameter:

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

`camdl simulate --params FILE` and `camdl pfilter --params FILE` read it, and
`--params` may be repeated: a later file overrides an earlier one, with a note
naming the value it replaced. A value must be a number; a string such as
`beta = "0.3 /day"` is refused with an error naming the file and key. A
parameter that no source supplies is an error naming it. The table-section
form for indexed parameters, the `[provenance]` table, and the other commands
that read the same format are specified in the run spec (Appendix B).

### 20.2 Priors

A prior is written with `~` on the parameter's declaration, after its kind and
bounds. It is optional.

```camdl
parameters {
    beta  : rate in [0.01, 2.0] ~ log_normal(mu = -1.0, sigma = 0.5)
    gamma : rate in [0.05, 1.0] ~ half_normal(sigma = 0.3)
    rho   : probability in [0.001, 1.0] ~ beta(alpha = 2.0, beta = 5.0)
    N0    : count in [100, 1_000_000]  # no prior — must be supplied via
                                       # --params, --scenario, or [fixed] in fit.toml
}
```

A parameter without a prior can still be given a value (a params file, a
scenario, a fit's `[fixed]` table) or be estimated; which methods require a
prior is stated in the run spec (§12.2). A prior in `fit.toml`'s `[estimate]`
overrides the model's (run spec §12.1); the `fit.toml` spellings are in run
spec §12.3.

**Families.** Nine families are accepted, each with keyword arguments only:

| Distribution       | Syntax                                 | Arguments                                         |
| ------------------ | -------------------------------------- | ------------------------------------------------- |
| `uniform`          | `~ uniform(lower = L, upper = U)`      | bounds, `L < U`                                   |
| `normal`           | `~ normal(mu = M, sigma = S)`          | mean and standard deviation, on the natural scale |
| `log_normal`       | `~ log_normal(mu = M, sigma = S)`      | mean and standard deviation of `log X`            |
| `half_normal`      | `~ half_normal(sigma = S)`             | standard deviation of the unfolded normal         |
| `beta`             | `~ beta(alpha = A, beta = B)`          | shape parameters                                  |
| `gamma`            | `~ gamma(shape = K, rate = R)`         | shape and rate, `E[X] = K/R`                      |
| `exponential`      | `~ exponential(rate = R)`              | rate, `E[X] = 1/R`                                |
| `log_uniform`      | `~ log_uniform(lower = L, upper = U)`  | bounds, `0 < L < U`                               |
| `truncated_normal` | `~ truncated_normal(mean = M, sd = S)` | mean and standard deviation; bounds from `in [..]` |

**Meaning.** The conventions that differ between texts are fixed as follows.

- `log_normal(mu, sigma)` takes its arguments on the log scale:
  `log(X) ~ Normal(mu, sigma)`, so the median of `X` is `exp(mu)`.
- `gamma(shape, rate)` is the rate parameterization, with density
  `rᵏ xᵏ⁻¹ e^(−r x) / Γ(k)`; a `scale` argument is refused.
- `log_uniform(lower, upper)` is uniform on the log scale,
  `log(X) ~ Uniform(log lower, log upper)`, with density
  `1 / (x (ln U − ln L))` on `[L, U]`: each order of magnitude in the range
  has equal mass. It suits a scale parameter known only to within orders of
  magnitude, such as a coupling rate.
- `truncated_normal(mean, sd)` is `Normal(mean, sd)` truncated to the
  parameter's declared `in [lo, hi]` and renormalized over it, so the bounds
  are the truncation and are stated in one place.

**Hyperparameters.** A prior argument may be an expression over other declared
parameters, which makes the prior hierarchical: each cell of an indexed
parameter is drawn from the same family with shared hyperparameters, which
pools the cells partially. An optional `| dim` suffix names the dimension
pooled over:

```
parameters {
  mu_h      : real ~ normal(mu = -1, sigma = 1)
  sigma_h   : positive ~ half_normal(sigma = 0.5)
  beta[age] : rate ~ log_normal(mu = mu_h, sigma = sigma_h) | age
}
```

A hierarchical prior's arguments are evaluated at each parameter draw, before
any simulation, so they may use parameters, literals and arithmetic on them.

**What is refused.**

- A positional argument is a syntax error (**E001**); an unknown family is
  **E232**; a missing argument is **E231**; an argument the family does not take
  (`normal(mean = …)`, `gamma(scale = …)`) is **E233**.
- A constant argument outside the family's domain (`uniform` with
  `lower ≥ upper`, `log_uniform` with `lower ≤ 0`, a non-positive `sigma`) is
  **E235**.
- A `truncated_normal` prior on a parameter with no `in [lo, hi]` is **E285**.
- An argument that reads time, compartment state, a `let` binding, a forcing, a
  table or an observation, or names an undeclared parameter, is **E230**.
- `log_uniform` and `truncated_normal` take constant arguments only; a
  hyperparameter in their arguments, or a `| dim` suffix, is **E286**.

Whether a prior puts mass on the whole of a parameter's search box (for
example, `log_uniform(lower = L, upper = U)` on bounds outside `[L, U]`, or a
`beta` prior on a `rate`) is not checked when the model is compiled. It is
checked when a fit starts (run spec §12.4).

Related sections: §4 (parameter declarations and bounds), §4.4 (bounds),
run spec §12 (prior precedence and the `fit.toml` forms).

### 20.3 Estimated and fixed parameters

A fit partitions the model's parameters into those it estimates and those it
holds fixed. The partition is the `fit.toml`'s `[estimate]` and `[fixed]`
tables (run spec §6.3, §6.4); `camdl profile` and `camdl survey` also take
`--fixed NAME=VALUE`. Forward simulation has no partition: every parameter
takes a value (§20.1).

### 20.4 Relationship to the Parameter Grammar

The parameter grammar (Buffalo 2026) is a formal framework for partitioning and
manipulating a simulator's inputs. Each of its objects has a camdl counterpart:

| Grammar concept          | camdl counterpart                                                                 |
| ------------------------ | --------------------------------------------------------------------------------- |
| **M** (parameter space)  | the `parameters { }` block                                                        |
| **C** (configuration)    | the model structure, `simulate { }` and `output { }`, and the backend and step chosen at run time |
| **S** (seed)             | `--seed` on the command line; never in the model file                             |
| **Point m ∈ M**          | a params file (§20.1)                                                             |
| **Scenario σ**           | a scenario in `scenarios { }`, a patch (§17)                                      |
| **Baseline σ₀**          | no scenario selected: the model as declared                                       |
| **View V**               | the `[estimate]` and `[fixed]` tables of a `fit.toml` (§20.3)                      |
| **Prior π(m)**           | `~` on a parameter declaration (§20.2)                                            |
| **Transform T_V**        | derived per estimated parameter from its resolved bounds and declared kind (run spec §6.3) |
| **Reparameterization R** | not supported as a separate input; a different parameterization is written in the model |
| **Sim(m, c, s) → y**     | `camdl simulate`                                                                  |
| **Sim_σ,V,T(z, s) → y**  | `camdl fit run`                                                                   |

The chain from inference coordinates to simulation output:

```
z ∈ Z_V     inference engine proposes a vector
  │ T_V⁻¹   inverse transform (run spec §6.3)
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

The model file declares the parameter space, the priors and the scenarios. The
point, the view, the scenario selected and the seed are chosen outside it.

---

## 21. CLI

The `camdl` command compiles, simulates and fits models. Every command that
takes a model accepts a `.camdl` file, which it compiles with `camdlc`. This
section lists the commands that act on the language's constructs; the complete
reference for every flag is the run spec (Appendix A), and fitting is described
in `camdl docs inference`.

### 21.1 Compilation and Inspection

```bash
camdl dev compile MODEL.camdl          # compile to IR JSON (stdout)
camdl check   MODEL.camdl              # parse and type-check; print diagnostics
camdl inspect MODEL.camdl [OPTIONS]    # inspect compartments, transitions, etc.
```

`check`, `inspect`, `render` and `dev compile` pass their arguments to
`camdlc`. `inspect` takes `--summary`, `--dims`, `--compartments`,
`--transitions`, `--tables` and `--forcings`; `camdlc --help` lists the
compiler's own flags.

### 21.2 Simulation

```bash
camdl simulate MODEL --params P.toml --seed 42 [OPTIONS]

Options:
  --backend    gillespie|chain_binomial|ode  (default: chain_binomial)
  --dt         DT         step size for chain_binomial / ode (default: 1.0)
  --seed       N          RNG seed (default: 1)
  --scenario   NAME       select a named scenario (repeatable; conflicts with --enable/--disable)
  --enable     NAME       enable an intervention (repeatable)
  --disable    NAME       disable an intervention (repeatable)
  --param      NAME=VALUE override a parameter value
  --param-vec  PREFIX=FILE override indexed params from a keyed TSV
  --params     FILE.toml  load parameter values (repeatable, later overrides earlier)
```

`simulate` writes the trajectory to a content-addressed store leaf under
`./results` (`--output-dir` sets the root), and prints the `camdl cat` command
that reads it back. `--stdout` streams the TSV to stdout and writes no leaf;
`-o FILE` writes a TSV copy in addition to the leaf. The trajectory's columns
are `t`, `t_start` and `t_stop` (the period each row's flows cover), one column
per compartment, and `flow_<name>` per transition (run spec §10.1).
Observation draws are written with `--obs`, `--obs-dir` or `--obs-only` (§12.4).

### 21.3 Expression Evaluation

```bash
camdl dev eval MODEL --params P.toml --expr "school,seas,R0" --from 0 --to 365 --every 1
camdl dev eval MODEL --params P.toml --expr "school" --at 0,100,200,300
```

`dev eval` evaluates forcings and parameters, by name, on a time grid without
simulating, and writes one column per name. A name that is not a forcing or a
parameter is refused with a message listing the names available. A `let`
binding is refused, since it is inlined into the expressions that use it, and so
is a compartment, whose value needs a simulation.

### 21.4 Sweeps and scenario grids

```bash
camdl simulate MODEL --params P.toml --sweep PARAM=GRID [--sweep PARAM=GRID …] \
    [--scenario A,B] [--replicates N | --seeds SPEC] [--obs-dir DIR] [--force] [--dry-run]
camdl simulate MODEL --params P.toml --sweep-file DESIGN.tsv [--replicates N]
camdl list              [RESULTS_DIR]          # browse cached runs
camdl show  <short-hash>
camdl cat   <short-hash> [--stream NAME]
```

`--sweep` runs a deterministic grid over parameters (the design axis),
`--scenario` a list of scenarios, and `--draws` parameter draws for a
predictive check. Every design point at a given replicate shares one random
stream, so the grid runs with common random numbers, and a second run of the
same grid reads back the cells the store already holds. The semantics are in
the run spec (§5).

camdl computes no sensitivity indices and generates no space-filling designs.
For a Sobol analysis, generate the design in R or Python, run it with
`--sweep-file`, and compute the indices from `ensemble.tsv` with R's
`sensitivity` package (Iooss et al., [CRAN](https://CRAN.R-project.org/package=sensitivity))
or Python's SALib (Herman and Usher 2017,
[_Journal of Open Source Software_ 2(9):97](https://doi.org/10.21105/joss.00097)).

### 21.5 Inference

```bash
# Particle filter: a log-likelihood estimate at fixed parameters
camdl pfilter MODEL --params P.toml --data cases.tsv \
    --particles 5000 --dt 1 --seed 42 --trace diag.tsv

# A fit: the [method] of a fit.toml (`camdl fit methods` lists them)
camdl fit run fit.toml --seed 42

# Profile likelihood over a grid of one parameter (repeat --sweep for 2D+)
camdl profile MODEL --starts from_params=P.toml --data cases.tsv \
    --sweep "R0=lin(0.5,5,20)" \
    --rw-sd "sigma=0.01,gamma=0.01" \
    --particles 500 --iterations 30 --n-starts 3 --parallel 8
```

The projection and the distribution of each data stream come from the model's
`observations { }` block (§12); no inference command overrides them. A bare
`--data FILE` binds to the model's only observation stream; when the model
declares more than one, `--obs NAME` names the stream or indexed family the
file is. `--data NAME=PATH`, repeated, binds several streams at once.

A fit's configuration (`[estimate]`, `[fixed]`, `[data]`, `[method]` and the
chain `starts`) is specified in the run spec (§6), with an IF2 example in §8.1.
Initial-condition parameters such as `S0` and `I0` are estimated like any other
parameter, by listing them under `[estimate]`.

**`--rw-sd`** (`camdl profile`) sets the IF2 perturbation scale of each
estimated parameter, and with it the estimated set:

- `--rw-sd "R0=5,sigma=0.01"`: the listed parameters are estimated, with these
  scales on the natural scale; the others are held fixed.
- `--rw-sd auto`: every parameter that is not swept, pinned (`--fixed`,
  `--fixed-file`, the `--fit` config's `[fixed]`) or a model constant is
  estimated, each with the default step that its bounds' transform takes.
- `--rw-sd "R0=5,sigma=auto"`: an explicit scale where one is given and the
  default step where `auto` is written.

`--rw-sd` is required without `--fit`. With `--fit`, it is optional: the
estimated set is the config's `[estimate]` less the swept and pinned
parameters, each with its `rw_sd` or the default step, and a list given with
`--fit` must name every such parameter. In a fit, IF2 takes each parameter's
scale from its `[estimate]` `rw_sd`, or the default step when `rw_sd` is
absent. Cooling follows pomp's `cooling.fraction.50`: the perturbation SD
reaches `cooling` times its initial value after `cooling_target_iters`
iterations (default 50; `docs/methods/cooling.md`).

**`--fixed NAME=VALUE`** (`camdl profile`, `camdl survey`) pins `NAME` at
`VALUE` and removes it from the estimated set; it may be repeated.
`--fixed-file FILE` (`camdl profile`) pins every key of a params file (§20.1),
and may be repeated, a later file overriding an earlier one. A parameter named
by neither takes its value from the next tier of the precedence chain (run spec
§1.3); a parameter with no value from any tier is an error. In a `fit.toml` the
equivalent is the `[fixed]` table.

### 21.6 Fit Workflow

```bash
camdl fit run     fit.toml [--starts SPEC] [--seed N] [--force] [--sweep "PARAM=V1,V2,..."]
camdl fit summary results/fits/<dir>/
camdl fit table   results/fits/
```

`fit run` runs the one `[method]` of a `fit.toml`. A sequence of fits, by
convention a scout, a refine and a validate file, is a sequence of files, each
starting from an earlier fit by handle in its `starts` rule
(`starts = { from_mle = "@scout" }`); `--starts SPEC` overrides the rule for one
run. A source fit whose convergence verdict failed is refused unless
`--allow-nonconverged-source` is passed. `--sweep` repeats the fit over a
Cartesian grid of fixed parameters (run spec §7.3). `fit summary` reports one
fit's convergence verdict, estimates and diagnostics; `fit table` writes one row
per fit under a results root. The method ladder is described in the inference
spec (§6).

**Particle-filter replicates:**

```bash
camdl pfilter MODEL --params P.toml --data d.tsv \
    --replicates 100 --output logliks.tsv
```

runs independent particle filters and writes a `seed`, `loglik` TSV, one row per
replicate.

**Particle-filter trace:**

```bash
camdl pfilter MODEL --params P.toml --data d.tsv --trace diag.tsv
```

writes one row per observation time, with columns
`time ll_increment ESS obs_mean obs_q05 obs_q50 obs_q95 state_mean state_q05
state_q50 state_q95 observed`. `ll_increment` is that time's contribution to
the log-likelihood and `ESS` the effective sample size of the particles. The
`obs_*` columns include observation noise; the `state_*` columns show process
uncertainty only.

### 21.7 Train/Holdout Validation

A train/holdout split is declared in the fit config, and the data file stays
whole. Under `holdout_after`, a row trains when the end of the period it
measures is at or before the cutoff:

```toml
[data]
holdout_after = 5474   # rows ending after t = 5474 are withheld from training

[data.observations]
cases = "data/cases.tsv"
```

Held-out rows supplied as their own file are bound with `[data.holdout]`
instead. `camdl compare` scores the withheld rows out of sample, and
`camdl pfilter --fit fit.toml --save-final-state FILE` binds only the training
rows and saves the particle state at their end. The rules for the cutoff are in
`docs/fit-toml.md`.

### 21.8 Particle State Export

```bash
camdl pfilter MODEL --params P.toml --data train.tsv \
    --particles 5000 --save-final-state final_particles.tsv
```

writes the final particle states to a TSV.

---

## 22. Worked Examples

These examples build from a bare SIR model to stratified and staged models,
each showing how the primitives compose and what the compiler expands them to.

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

Two transitions with unindexed rate expressions; no stratification, demography
or observations.

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

Six transitions. Every rate is a total propensity: the `death_S` rate is
`mu * S`, a per-capita rate times a count. `birth` is an inflow at `mu * N`,
which balances the deaths in expectation.

### 22.3 SEIR with Age Mixing (Introducing Stratification)

Age-structured transmission is written in the explicit indexed form (§10): one
`infection` transition per age class, with a contact-matrix-weighted sum over
the infectious classes.

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

`sum(b in age, C_age[a,b] * I[b] / N_local[b])` is the
contact-matrix-weighted force of infection on age class `a`. The `[a in age]`
binder replicates progression and recovery within each stratum, so the model
expands to six transitions.

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

The zero diagonal of `B_sex` removes within-sex transmission. The sum over
`t in sex` keeps both terms, and the zero entry weights the same-sex term to
zero, so the rate of `infection_female` equals
`S[female] * beta_mf * I[male] / N_local[male]`. Directed transmission needs no
keyword of its own; the matrix states it.

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

`W : real` is a continuous quantity, not a population count, and the `ode`
block gives its derivative `dW/dt`. W evolves deterministically between the
stochastic infection and recovery events (§11), and enters the infection rate
through the dose-response term `beta_W * W / (K + W)`. `c in compartments`
iterates over integer compartments only, so it does not include `W` (§9.5).

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
  beta  : probability   # per contact; C_age carries the per-day contact rate
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

`consecutive(age)` generates the pairs `(age_0_5, age_5_15)`,
`(age_5_15, age_15_50)`, `(age_15_50, age_50_65)` and `(age_50_65, age_65p)`.
With 3 compartments, `aging` expands to 3 × 4 = 12 transitions. The last age
group, `age_65p`, has no outgoing aging; individuals leave it only by death.
The model has 5 infections + 5 recoveries + 12 aging + 15 deaths + 1 birth = 38
transitions.

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

An Erlang-3 latent period has the same mean as the exponential, `1/sigma`, and
a smaller variance, `1/(k·sigma²)` with `k = 3`, so the distribution is more
concentrated about its mean. `infection` enters the first sub-stage, `E[e1]`.
Under partial stratification (`only = [E]`), S, I and R do not carry the
`latent_stage` dimension. The same residence is written in one line as
`onset : E --> I via erlang(stages = 3, rate = sigma)` (§9.4.1), under which
the stages are residence structure and not a population stratum (§5.0).

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
  import_rate : positive dim P/T   # importations per unit time
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
    covers        = closing_at(time, 7 'days)
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

`camdlc` compiles a `.camdl` file into the intermediate representation (IR), a
JSON document that the `camdl` runtime reads. The runtime does not read the
model source: it sees only the IR's concrete compartments, transitions and
expression trees, with no index variables left in them. The IR format is
specified in `docs/compartmental-ir-spec.md`.

```
model.camdl ─▶ parse ─▶ expand ─▶ check ─▶ model.ir.json
                          ▲
             files named by read(…)

model.ir.json + parameter values + data ─▶ camdl ─▶ simulations, fits, output
```

- **Parse.** The text becomes a syntax tree. A syntax error is **E001**.
- **Expand.** Every indexed declaration becomes the concrete compartments,
  transitions, observations and actions it stands for (§25): binders are
  enumerated, `where` guards are evaluated and the members they reject are
  dropped, `let` bindings are lowered, unit literals are converted to the
  model's `time_unit`, and the files named by `read(…)` are read (§6.2).
- **Check.** Names, index arity, table shapes and dimensions (§2.2) are checked,
  during expansion and on the expanded model.

`camdlc check FILE.camdl` runs all three stages and writes nothing.
`camdlc FILE.camdl` writes the IR to standard output, one element per line;
`--pretty` indents it, and `--set NAME=VALUE` overrides a parameter's value in
it. Parameter values (§20), observation data and run settings are otherwise read
by the runtime when a model is run (§21).

### 24.1 File-Level Grammar

A `.camdl` file is a sequence of declarations, optionally headed by a `#'`
block that documents the model (§1.1).

```camdl
file := model_doc? declaration*

declaration :=
  | time_unit_decl                    # time_unit = 'days
  | description_decl                  # description = "..."
  | origin_decl                       # origin = date("YYYY-MM-DD")
  | compartments_block                # compartments { ... }
  | parameters_block                  # parameters { ... }
  | dimensions_block                  # dimensions { dim = [...] | read(...) }
  | stratify_decl                     # stratify(by = ..., ...)
  | tables_block                      # tables { ... }
  | forcing_block                     # forcing { ... }
  | let_binding                       # let NAME = EXPR  |  let NAME : TYPE = EXPR
  | transitions_block                 # transitions { ... }
  | ode_block                         # ode { ... }
  | balance_block                     # balance { NAME = EXPR }
  | init_block                        # init { ... }
  | observations_block                # observations { ... }
  | interventions_block               # interventions { ... }
  | events_block                      # events { ... }
  | reactive_interventions_block      # reactive_interventions { ... }
  | timepoints_block                  # timepoints { ... }
  | quantities_block                  # quantities { ... }
  | contrasts_block                   # contrasts { ... }
  | output_block                      # output { ... }
  | simulate_block                    # simulate { ... }
  | scenarios_block                   # scenarios { ... }
```

Declarations may appear in any order. All of them are collected before any name
is resolved, so a name may be used above its declaration. A block that lists
entries (`compartments`, `parameters`, `transitions`, …) may be written more than
once, and its entries are joined. `time_unit`, `origin`, `description`,
`simulate`, `output` and `balance` may each be written once; a second one is
**E140**, located at the second, with a note at the first.

The only declaration a model requires is `compartments`. A model with no other
declaration compiles, with an L402 warning for each compartment that nothing
reads, and runs. Each omitted declaration has a default: `time_unit` is
`'days` (§2), the run covers model times 0 to 100 when there is no `simulate`
block (§14.1), a compartment that `init` does not name starts at 0 (§15), and a
model with neither `transitions` nor `ode` holds its initial state.

Related sections: §25 (what expansion produces), §26 (diagnostics), §21 (the
commands that compile and run a model).

---

## 25. Expansion Rules (DSL → IR Mapping)

This section gives the IR that each construct compiles to, as `camdlc` writes
it. The IR fragments are abridged: a transition's `metadata`, `rate_grad` and
`rate_state_grad` fields are left out, and `...` stands for elements of the
same pattern.

Expanded names join a base name and levels with `_`. A compartment cell takes
its levels in the order the dimensions were stratified (`S[child, p1]` is
`S_child_p1`); a member of a transition family takes the level of each binder,
in binder order (`migrate[a in age, src in patch, dst in patch]` has a member
`migrate_child_p1_p2`). The examples use `age = [child, adult]` unless they say
otherwise.

### 25.1 Let Bindings

A `let` lowers to one of three forms:

- a typed `let` whose body is a constant is a model constant, listed under
  `constants` and read through `const_ref`;
- a `let` whose body reads no parameter, directly or through another `let`, is a
  shared binding, listed once per cell under `bindings` and read through
  `binding_ref`;
- any other `let` is inlined: its expression tree replaces each use.

```camdl
# DSL:
let N = S + E + I + R             # reads no parameter
let k : positive = 2.0            # typed, with a constant body
let force = beta * I / N          # reads a parameter

# IR:
"constants": [{"name": "k", "value": 2.0, "kind": "positive"}]
"bindings":  [{"name": "N", "expr": {"pop_sum": ["S_child", "S_adult", "E_child", "E_adult",
                                                  "I_child", "I_adult", "R_child", "R_adult"]}}]
# a use of k:     {"const_ref": "k"}
# a use of N:     {"binding_ref": "N"}
# a use of force: {"bin_op": {"op": "div",
#                   "left": {"bin_op": {"op": "mul", "left": {"param": "beta"},
#                                       "right": {"pop_sum": ["I_child", "I_adult"]}}},
#                   "right": {"binding_ref": "N"}}}
```

In model dynamics a bare stratified compartment name is the sum over all its
cells (`pop_sum`), so `N` is the total population. An observation reads the bare
name differently (§12.1, "Pooling strata"). An indexed `let` that reads no
parameter, such as `let N_local[a in age] = S[a] + E[a] + I[a] + R[a]`, gives
one binding per cell: `N_local_child` and `N_local_adult`.

### 25.2 Indexed Transitions

An indexed transition becomes one IR transition per combination of its binders'
levels.

```camdl
# DSL:
recovery[a in age] : I[a] --> R[a]  @ gamma * I[a]

# IR: two transitions
{"name": "recovery_child",
 "stoichiometry": [["I_child", -1], ["R_child", 1]],
 "rate": {"bin_op": {"op": "mul", "left": {"param": "gamma"}, "right": {"pop": "I_child"}}}}

{"name": "recovery_adult",
 "stoichiometry": [["I_adult", -1], ["R_adult", 1]],
 "rate": {"bin_op": {"op": "mul", "left": {"param": "gamma"}, "right": {"pop": "I_adult"}}}}
```

### 25.3 Inflows

An inflow has a stoichiometry with only a destination.

```camdl
# DSL, with patch = [p1, p2, p3]:
birth[p in patch] : --> S[child, p]
  @ mu * sum(a in age, N_local[a, p])

# IR, for p = p1:
{"name": "birth_p1",
 "stoichiometry": [["S_child_p1", 1]],
 "rate": {"bin_op": {"op": "mul", "left": {"param": "mu"},
   "right": {"reduce": [{"binding_ref": "N_local_child_p1"},
                        {"binding_ref": "N_local_adult_p1"}]}}}}
```

A `sum` over a dimension lowers to `reduce`, the sum of one term per level. Here
each term is a cell of the shared binding `N_local` (§25.1); written out as
`sum(a in age, S[a, p] + E[a, p] + I[a, p] + R[a, p])`, the same sum lowers to
one `pop_sum` over the cells of patch `p1`.

### 25.4 Projections

An observation's projection (§12.1) lowers to one of four IR forms: a single
flow, a list of flows accumulated together, a ratio of two such lists, or a sum
of compartment cells read at an instant.

```camdl
# DSL, on an unstratified `infection`:
incidence(infection)

# IR:
{"cumulative_flow": "infection"}

# DSL: one stratum of a stratified family
incidence(infection[child])

# IR:
{"cumulative_flow": "infection_child"}

# DSL: a stratified family pooled with `sum`, and flows added into one column
sum(a in age, incidence(infection[a]))
incidence(infection[child]) + incidence(infection[adult])
sum(a in age, incidence(infection[a])) + incidence(importation)

# IR:
{"cumulative_flow_sum": ["infection_child", "infection_adult"]}
{"cumulative_flow_sum": ["infection_child", "infection_adult"]}
{"cumulative_flow_sum": ["infection_child", "infection_adult", "importation"]}

# DSL: one sum of flows divided by another, both accumulated over the row's period
incidence(die_comm) / (incidence(die_comm) + incidence(die_fac))
incidence(infection[child]) / sum(a in age, incidence(infection[a]))

# IR:
{"flow_ratio": {"numerator": ["die_comm"], "denominator": ["die_comm", "die_fac"]}}
{"flow_ratio": {"numerator": ["infection_child"], "denominator": ["infection_child", "infection_adult"]}}

# DSL: a stock, pooled over its strata
sum(R)

# IR:
{"current_pop_sum": ["R_child", "R_adult"]}
```

Adding flows and summing a family produce the same object, a list of flows
accumulated over the row's period, so the two forms combine into one list. The
same flow on both sides of a ratio is the expected shape. Weighting, subtracting
or mixing flows with a state read is **E341**, and an unindexed `incidence(…)`
or compartment over a stratified family is **E280** (§12.1).

### 25.5 Interventions

A scheduled intervention lowers to its firing times and a list of actions. An
action on a stratified compartment becomes one action per cell.

```camdl
# DSL:
sia_round_1 : transfer(fraction = 0.80, from = S, to = V) at [180]

# IR:
{"name": "sia_round_1",
 "fire": {"scheduled": {"at_times": [180.0]}},
 "actions": [
   {"fraction_transfer": {"src": "S_child", "dst": "V_child", "fraction": {"const": 0.8}}},
   {"fraction_transfer": {"src": "S_adult", "dst": "V_adult", "fraction": {"const": 0.8}}}
 ]}
```

A `fraction_transfer` moves `⌊f · x⌋` individuals from the source cell to the
destination cell, where `x` is the source's current count and `f` is the
fraction clamped to `[0, 1]`. For `real` compartments, and under the ODE
backend, the amount `f · x` is not rounded. Interventions are off unless
enabled (§13.4), and the order in which actions and other effects apply within a
step is §13.10.

### 25.6 Stratified Transmission (Indexed Sum → IR)

The explicit transmission form of §10 lowers like any other indexed rate: the
inner `sum` becomes a `reduce`, and a table cell whose value is a literal number
is written as that number.

```camdl
# DSL, with C_age = [[2.0, 0.5], [0.5, 1.5]] and
# let N_local[a in age] = S[a] + E[a] + I[a] + R[a]:
infection[a in age] : S[a] --> E[a]
  @ beta * S[a] * sum(b in age, C_age[a, b] * I[b] / N_local[b])

# IR, for a = child:
{"name": "infection_child",
 "stoichiometry": [["S_child", -1], ["E_child", 1]],
 "rate": {"bin_op": {"op": "mul",
   "left": {"bin_op": {"op": "mul", "left": {"param": "beta"}, "right": {"pop": "S_child"}}},
   "right": {"reduce": [
     {"bin_op": {"op": "div",
       "left": {"bin_op": {"op": "mul", "left": {"const": 2.0}, "right": {"pop": "I_child"}}},
       "right": {"binding_ref": "N_local_child"}}},
     {"bin_op": {"op": "div",
       "left": {"bin_op": {"op": "mul", "left": {"const": 0.5}, "right": {"pop": "I_adult"}}},
       "right": {"binding_ref": "N_local_adult"}}}]}}}}
```

A cell whose value is an expression, such as a parameter, is read through
`table_lookup` at a constant row-major index instead.

### 25.7 Consecutive Pairs

A `consecutive(dim)` binder enumerates each level with the next one; the last
level has no successor and generates no transition.

```camdl
# DSL, with age = [age_0_5, age_5_15, age_15_50], compartments = [S, I, R]
# and age_dur : age 'days = [5 'years, 10 'years, 35 'years]:
aging[c in compartments, (a, a_next) in consecutive(age)]
  : c[a] --> c[a_next]
  @ (1 / age_dur[a]) * c[a]

# 2 pairs × 3 compartments = 6 IR transitions:
{"name": "aging_S_age_0_5",
 "stoichiometry": [["S_age_0_5", -1], ["S_age_5_15", 1]],
 "rate": {"bin_op": {"op": "mul",
   "left": {"bin_op": {"op": "div", "left": {"const": 1.0}, "right": {"const": 1826.2125}}},
   "right": {"pop": "S_age_0_5"}}}}

{"name": "aging_S_age_5_15",
 "stoichiometry": [["S_age_5_15", -1], ["S_age_15_50", 1]],
 "rate": {"bin_op": {"op": "mul",
   "left": {"bin_op": {"op": "div", "left": {"const": 1.0}, "right": {"const": 3652.425}}},
   "right": {"pop": "S_age_5_15"}}}}

# ... the same pattern for I and R
```

Both `a` and `a_next` may appear in the rate. The member's name carries `a` and
not `a_next`. The table's durations appear converted to the model's
`time_unit` (5 years is 1826.2125 days).

### 25.8 Guard Clauses (`where`)

A `where` guard is evaluated when the model is compiled. Each combination of
binder levels is instantiated, and a combination for which the guard is false
generates no transition. The IR has no guards.

```camdl
# DSL, with patch = [p1, p2, p3]:
migrate[src in patch, dst in patch] : S[src] --> S[dst]
  @ mig[dst,src] * S[src]
  where src != dst

# Combinations:
#   (p1, p1): src == dst → dropped
#   (p1, p2): src != dst → migrate_p1_p2
#   (p1, p3): src != dst → migrate_p1_p3
#   (p2, p1): src != dst → migrate_p2_p1
#   (p2, p2): src == dst → dropped
#   ...
# Result: 6 transitions of the 9 combinations
```

`camdlc inspect` reports the number of members a guard dropped (§26.11).

### 25.9 Compartment Iteration (`c in compartments`)

`c in compartments` substitutes each integer compartment name for `c`. When a
compartment has more dimensions than the index signature gives, the compiler
iterates over the omitted dimensions, so that every source and destination names
a full cell (§5.1).

```camdl
# DSL:
death[c in compartments, a in age] : c[a] -->  @ mu * c[a]
# with compartments = [S, I, R], R stratified by an extra dimension
# immunity = [natural, vaccine]
# S dims: [age], R dims: [age, immunity]

# For S:
{"name": "death_S_child", "stoichiometry": [["S_child", -1]],
 "rate": {"bin_op": {"op": "mul", "left": {"param": "mu"}, "right": {"pop": "S_child"}}}}
{"name": "death_S_adult", "stoichiometry": [["S_adult", -1]],
 "rate": {"bin_op": {"op": "mul", "left": {"param": "mu"}, "right": {"pop": "S_adult"}}}}

# For R, one transition per immunity level:
{"name": "death_R_child_natural", "stoichiometry": [["R_child_natural", -1]],
 "rate": {"bin_op": {"op": "mul", "left": {"param": "mu"}, "right": {"pop": "R_child_natural"}}}}
{"name": "death_R_child_vaccine", "stoichiometry": [["R_child_vaccine", -1]],
 "rate": {"bin_op": {"op": "mul", "left": {"param": "mu"}, "right": {"pop": "R_child_vaccine"}}}}
# ... (adult × natural, adult × vaccine)
```

### 25.10 Interventions (All Dimensions)

An action on a bare stratified compartment expands over all of its dimensions.

```camdl
# DSL:
sia_round_1 : transfer(fraction = 0.80, from = S, to = V) at [180]
# S and V stratified by [age, patch], 2 × 774 cells

# IR: one fraction_transfer per (age, patch) cell, 1548 in all
{"name": "sia_round_1",
 "fire": {"scheduled": {"at_times": [180.0]}},
 "actions": [
   {"fraction_transfer": {"src": "S_child_p1", "dst": "V_child_p1", "fraction": {"const": 0.8}}},
   {"fraction_transfer": {"src": "S_child_p2", "dst": "V_child_p2", "fraction": {"const": 0.8}}},
   ...
   {"fraction_transfer": {"src": "S_adult_p774", "dst": "V_adult_p774", "fraction": {"const": 0.8}}}
 ]}
```

The two endpoints pair cell by cell only when both are declared with the same
dimensions in the same order; anything else is **E237**. `count =` on this bare
form is **E238**, since it would move the count out of every cell, and a bare
endpoint inside an indexed intervention family is **E239** (all three: §13.1).

---

## 26. Errors and Validation

`camdlc` reports the problems in a model when it is compiled. Each diagnostic
has a code, a severity, a location and, for most, a hint. Parameter values are
checked when they are supplied for a run (§26.7).

### 26.0 Diagnostic Codes

Every diagnostic carries a code. The table below lists a selection. The leading
digit of an error code gives its family: `E001` (syntax), `E1xx` (names,
indices, reserved identifiers and singleton declarations), `E2xx` (declarations,
expansion, schedules and observation streams), `E3xx` (dimensions and times),
`E4xx` (forcing, constant expressions and declaration families), `E5xx` (names
in the expanded model, such as an expanded compartment that does not exist) and
`E6xx` (checks on the expanded model as a whole, such as the `simulate`
horizon). `W` codes are warnings, `L` codes are warnings about the model's
structure (L402, a compartment declared but never used), and `I` codes are
information.

| Code | Kind    | Description                                                                                |
| ---- | ------- | ----------------------------------------------------------------------------------------- |
| E001 | Error   | Parse / lexer syntax error (including using a keyword as a name)                           |
| E100 | Error   | Reserved name used as a declaration; unknown index value; undeclared name/function         |
| E101 | Error   | `origin` is not `date("YYYY-MM-DD")`                                                       |
| E104 | Error   | Unknown likelihood family in an observation                                                |
| E118 | Error   | The removed `likelihood = <value>` form, with a value that is not a distribution call      |
| E120 | Error   | A built-in function or a forcing called with the wrong number of arguments                 |
| E121 | Error   | An observation block assigns a value under a key other than `projected` (§12) |
| E122 | Error   | An observation block gives the same key (`columns`, `covers`, `projected`, `emit_schedule`, or the `~` line) twice (§12) |
| E123 | Error   | A scenario `enable`/`disable`/`compose` value is not a bracketed list of names — write `enable = [NAME]` (§17.1) |
| E124 | Error   | A scenario field other than `set`/`scale` written as a `{ ... }` block (§17.1) |
| E140 | Error   | A singleton declaration (`time_unit`, `description`, `origin`, `simulate`, `output`, `balance`) written twice |
| E200 | Error   | A `read()` data file is not found, or `read()` has no positional path argument             |
| E202 | Error   | Table arity / shape mismatch (e.g. `table '%s' expects %d indices`)                        |
| E228 | Error   | `time_unit` is not a duration unit (`'per_day` / `'count` / `'ratio`) — see §2               |
| E265 | Error   | Intervention `set`/`add` targets a name that is not one expanded compartment               |
| E272 | Error   | Removed observation cadence form — use `emit_schedule = …` (§12.4)                         |
| E278 | Error   | Duplicate declaration (a name declared more than once / in multiple namespaces)            |
| E300 | Error   | Transition rate has the wrong dimension                                                    |
| E302 | Error   | Addition/subtraction of mismatched dimensions                                             |
| E304 | Error   | A likelihood argument has the wrong dimension, or a square root has odd exponents          |
| E308 | Error   | Overdispersion σ² must have dimension T                                                    |
| E310 | Error   | A transition with no net effect: its sources and destinations cancel                       |
| E311 | Error   | A table lookup's index is not a pure number (an index selects a level)                     |
| E313 | Error   | An initial condition has the wrong dimension                                               |
| E314 | Error   | An intervention action's `fraction`, `count` or value has the wrong dimension              |
| E315 | Error   | An intervention or schedule time is not T, or a trigger threshold is not its stream's dimension |
| E316 | Error   | A unit literal in a prior argument or a bound disagrees with its dimension (§2.2.3)        |
| E317 | Error   | A quantity threshold is not its series' dimension, or a `value_at` time is not T           |
| E318 | Error   | A typed `let`'s body does not have its declared kind or `dim` clause (§8.4)                 |
| E319 | Error   | A dimension exponent too large to represent (a power of a power of a power …)              |
| I316 | Info    | A log-normal `mu` or `sigma` is directly a dimensioned parameter (§2.2.3)                  |
| E347 | Error   | Observation stream declares its temporal columns twice or by half — `: time` plus window columns, half a pair, or `covers` with window columns (§12.1.1) |
| E348 | Error   | `covers` or window columns on a stream whose `projected` reads state at an instant (§12.1.1) |
| E350 | Error   | Incidence stream states no period — add `covers = …` or `window_start`/`window_stop` columns (§12.1.1) |
| E351 | Error   | A side of a ratio of flows names no flow after its `where` guard pruned every level (§12.1) |
| E352 | Error   | A projection divides transition rates — an instant reading; write the flows or `prevalence(of = …, among = …)` (§12.1) |
| E356 | Error   | `prevalence(X)` / `prevalence(X1, X2)` are removed — write `X` for the count, or `prevalence(of = X, among = N)` for a proportion (§12.1) |
| E357 | Error   | `prevalence(of = …, among = …)`: the numerator reads a compartment cell the denominator does not (§12.1) |
| E358 | Error   | `prevalence(of = …, among = …)`: the two sides collapse different dimensions (§12.1) |
| E359 | Error   | `prevalence(of = …, among = …)` reads a flow on one side (§12.1) |
| E360 | Error   | Arithmetic with no meaning on a definite instant: `Instant + Instant`, a `sum` of instants, `k * Instant`, `Duration − Instant`, `Instant +` a unitless constant, a conditional choosing between an Instant and a duration (§2.1) |
| E377 | Error   | A stratified compartment is named without indices in a transition's stoichiometry (§8.3) |
| E378 | Error   | A reactive trigger's threshold is neither a constant nor a parameter (§13.9) |
| E379 | Error   | A `diagnostic_test` `sens` or `spec` that is a constant outside `[0, 1]` or not a finite number (§12.2.1) |
| E423 | Error   | A destination split's constant weights are not a probability distribution: a weight is not finite or lies outside [0, 1], or all are constant and do not sum to 1 (within `1e-9`) (§9.1.2) |
| W103 | Warning | Let binding name shadows a stratum value in some dimension                                |

`camdlc FILE.camdl --json-errors` writes the diagnostics to standard error as a
JSON array instead of as text. Each element has `severity`, `code`, `message`,
`loc` (file, line, column, end line and end column) and, where there is one,
`hint`. `camdlc check` does not take the flag.

```bash
camdlc model.camdl --json-errors 2>errors.json
```

### 26.1 Dimension Errors

An index on a compartment, table or `let` family names a level or a binder, one
per dimension. An index variable that no binder binds is read as a level name,
so `I[a, s]` with `s` unbound names the cell `I_child_s`, which does not exist,
and is **E100**. A binder of the wrong dimension in a table slot is **E263**,
naming each level that is not a level of the slot's dimension.

```camdl
# An index variable that is not bound (I : [age])
recovery[a in age] : I[a, s] --> R[a]  @ gamma * I[a, s]
# error[E100]: undeclared name 'I_child_s'

# A binder of the wrong dimension (C_age : age × age, j bound to sex)
infection[a in age] : S[a] --> E[a]
  @ beta * S[a] * sum(j in sex, C_age[a, j] * I[j] / N[j])
# error[E263]: 'f' is not a level of dimension 'age'
#   = hint: valid levels: child, adult
```

### 26.2 Unbound Variables

Every index variable in a rate is bound by the transition's index or by a
`sum`. An unbound one is **E100**, as in §26.1.

```camdl
infection[a in age] : S[a] --> E[a]
  @ beta * S[a] * I[a, s] / N
# with I : [age, sex] and s unbound:
# error[E100]: undeclared name 'I_child_s'
```

### 26.3 Partial Stratification Stoichiometry

Every source and destination of a transition names a full cell of its
compartment (§5.1). A destination that leaves out one of its compartment's
dimensions names no cell and is **E503**.

```camdl
dimensions { immunity = [natural, vaccine] }
stratify(by = immunity, only = [R])

recovery[a in age] : I[a] --> R[a]  @ gamma * I[a]
# R : [age, immunity], so R[a] names no cell:
# error[E503]: unknown compartment referenced: 'R_child'
# Write R[a, natural] or R[a, vaccine].
```

### 26.4 Dimension Does Not Exist

A binder over a name that is not a declared dimension is **E263**; the hint lists
the declared dimensions.

```camdl
recovery[a in age, r in habitat] : I[a, r] --> R[a, r]  @ gamma * I[a, r]
# error[E263]: 'habitat' is not a declared dimension
#   = hint: ... declared dimensions: age, sex, patch
```

### 26.5 Compartment Doesn't Have Dimension

An index of a dimension that the compartment is not stratified by names no cell
and is **E100**.

```camdl
dimensions { immunity = [natural, vaccine] }
stratify(by = immunity, only = [R])

waning[a in age] : I[a, natural] --> S[a]  @ wane * I[a, natural]
# I : [age], so I[a, natural] names no cell:
# error[E100]: undeclared name 'I_child_natural'
```

### 26.6 Unit Errors

The two sides of `+` or `-` have the same dimension (§2.2); a mismatch is
**E302**. A rate whose dimension is not a population per unit time is **E300**.

```camdl
transitions {
  recovery : I --> R  @ gamma + I
  # where gamma : rate ('per_day) and I : count
}
# error[E302]: dimension mismatch: + in transition 'recovery' rate
#   = note: left has dimension T^-1 (per-capita rate), right has dimension
#     P (population count) — cannot combine
```

### 26.7 Parameter Domain Errors

A parameter value is checked against the parameter's declared type and bounds
(§4.4) when parameter values are resolved for a run, not when the model is
compiled. A value outside them stops the run before it starts:

```
# params.toml: rho = 1.5, with rho : probability
error: parameter 'rho' = 1.5 is outside declared bounds [0, 1].
```

### 26.8 Scenario Validation

A parameter named twice in one scenario's `set` or `scale` block is an error.
Two scenarios composed with `compose` that both modify one parameter draw a
warning, because composition applies the scenarios in order and its result
depends on that order (§17).

```camdl
scenarios {
  high_coverage {
    scale = {
      beta = 1.5
      beta = 2.0
    }
  }
}
# error: parameter 'beta' appears twice in scale operation.

scenarios {
  combined {
    compose = [variant, closure]
  }
}
# warning: scenarios 'variant' and 'closure' both modify parameter 'beta';
#   'variant' is applied first, then 'closure'.
```

### 26.9 Self-Loop Detection

A transition whose sources and destinations cancel has no net effect and is
**E310**. A family whose binders include a member with the same source and
destination, such as the diagonal `src == dst` of a migration family, is refused
for that member.

```camdl
migrate[c in compartments, src in patch, dst in patch]
  : c[src] --> c[dst]  @ mig[dst, src] * c[src]
# error[E310]: transition 'migrate' has no net effect: sources and
#   destinations cancel
```

A `where` guard drops the diagonal (§9.3):

```camdl
migrate[c in compartments, src in patch, dst in patch]
  : c[src] --> c[dst]  @ mig[dst, src] * c[src]  where src != dst
```

### 26.10 Name Resolution

A user-defined name lives in exactly one namespace. A name declared twice, in
one namespace or in two of compartments, parameters, `let` bindings, forcing and
tables, is **E278**, so no reference is ambiguous and there is no resolution
order between namespaces. Two parameters named `beta`, two compartments named
`S`, and a parameter and a `let` both named `beta` are all **E278**.

A reserved identifier (`t`, `t_start`, `t_end`, `dt`, `pi`, `e`; §14.2) used as
the name of a parameter or `let` is **E100**. A keyword (`compartments`, `sum`,
`consecutive`, `origin`, …) used as a name is a syntax error, **E001**. A `let`
named like a level of a dimension is allowed, with warning **W103**.

### 26.11 Compiler Reporting

`camdlc inspect FILE.camdl` prints a summary of the model's size: each count as
declared and as expanded, and the number of transitions a `where` guard dropped.
The header is the model's name. For an SEIR model with a vaccinated class,
stratified by two age groups and three patches:

```
strat

  compartments   5 base × 2 age × 3 patch = 30 expanded
  transitions     5 base → 33 expanded (+ 6 filtered by where)
  parameters      5 declared (5 rate)
  tables          0
  let bindings    1 (N_local[a in age, p in patch])
  dimensions      age = [child, adult], patch = [p1, p2, p3]
  observations    0 streams
  interventions   1 (0 active by default)
```

`camdlc inspect` takes one report option in place of the summary, such as
`--parameters` for the parameters grouped by kind, `--compartments` for the
expanded names, or `--count` for the members each declaration generates;
`camdlc inspect --help` lists them all.

---

## 27. Primitive Summary

```camdl ignore
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
NAME                                 bare: in dynamics, the sum over all strata (§12.1 for observations)
[var in dim]                         bind index variable to dimension
sum(var in dim, expr)                sum over dimension
sum(var in dim where PRED, expr)     sum over the levels the guard keeps

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
table : dim × dim 'UNIT = [...]      typed, shape-checked, unit-annotated
let name[indices] = expr             computed quantity (family of values)
let name : TYPE = expr               typed let (§8.4)

# Math functions
exp(x), log(x), sqrt(x)              standard math (unary)
abs(x), floor(x), ceil(x)            rounding and absolute value
sin(x), cos(x), tanh(x)              trigonometric / hyperbolic (unary)
mod(a, b)                            Euclidean remainder (binary)
min(a, b), max(a, b)                 smaller / larger of two values

# Time
t                                    current simulation time

# Forcing (time-dependent functions); the unit literal 'UNIT is required (§7; E001 if omitted)
forcing { NAME : sinusoidal 'UNIT { ... } }       smooth seasonal
forcing { NAME : fourier 'UNIT { ... } }          finite Fourier series
forcing { NAME : periodic 'UNIT { ... } }         repeating step (values or on = [lo:hi, ...])
forcing { NAME : periodic_spline 'UNIT { ... } }  periodic B-spline
forcing { NAME : bspline 'UNIT { ... } }          non-periodic B-spline
forcing { NAME : piecewise 'UNIT { ... } }        non-repeating step
forcing { NAME : interpolated 'UNIT { ... } }     data-driven (linear or spline)

# Reserved identifiers
t, t_start, t_end, dt, pi, e                # reserved (E100 if used as a name)
compartments, sum, consecutive, origin      # keywords (E001 if used as a name)
```
