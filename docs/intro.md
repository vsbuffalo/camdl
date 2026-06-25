# camdl by Example

## What camdl is

camdl (Compartmental Model Description Language) describes stochastic
compartmental models. A `.camdl` file says what the compartments are, how
individuals move between them, and how fast. It compiles to a flat JSON
intermediate representation that a Rust backend simulates.

**Three design commitments:**

1. **Structure lives in the model, values live outside.** A `.camdl` file
   defines compartments, transitions, and parameter _names_. Parameter _values_
   come from external TOML files or CLI flags — never hardcoded. The same model
   file serves forward simulation, calibration, and scenario comparison.

2. **Everything expands.** Stratification, indexed parameters, indexed
   interventions — all expand to flat, concrete IR at compile time. The runtime
   sees no index variables, no dimension metadata, no sugar. If the model has
   238 patches × 5 compartments, the IR has 1,190 compartment entries with
   explicit names like `S_kano_dala`.

3. **Write the math.** The index notation `[a in age]` reads like mathematical
   subscripts. Transitions read as "from → to at rate." Tables are lookup
   arrays. The goal is that a modeler can read a `.camdl` file and know what it
   does without learning a programming language.

---

## A single transition

The simplest possible model: susceptible individuals become infected.

```camdl
time_unit = 'days

compartments { S, I }

parameters {
  lambda : rate
  N0     : count
}

transitions {
  infection : S --> I  @ lambda * S
}

init { S = N0 }

simulate {
  from = 0 'days
  to   = 30 'days
}
```

The `@` introduces the **propensity** — the total expected events per unit time
across the whole population. If there are 900 susceptibles and lambda =
0.01/day, the propensity is 9 events/day.

Run it:

```bash
camdl simulate model.camdl --params params.toml --seed 42
```

Where `params.toml`:

```toml
lambda = 0.01
N0 = 1000
```

Or override directly:

```bash
camdl simulate model.camdl --param lambda=0.01 --param N0=1000 --seed 42
```

---

## SIR: propensity = per-capita rate × population at risk

Recovery adds a second transition:

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
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}
```

Read each transition as: "from [source] to [destination] at total rate
[propensity]."

| Transition | Per-capita rate | × Who's at risk | = Propensity  |
| ---------- | --------------- | --------------- | ------------- |
| infection  | β × I / N       | S               | β × S × I / N |
| recovery   | γ               | I               | γ × I         |

The per-capita rate is the hazard each individual faces. The propensity is that
hazard times the number of individuals at risk. camdl always wants the
propensity — you write the multiplication explicitly. The compiler never
silently multiplies by a population count.

**`beta` is a rate. `beta * S * I / N` is a propensity.** This distinction
matters: rates have units of 1/time, propensities have units of events/time.
Every `@` expression is a propensity.

---

## How names resolve: parameters, let bindings, tables

Three ways to name things in camdl. They have different lifetimes and different
roles:

### Parameters: external knobs

```camdl
parameters {
  beta  : rate
  gamma : rate
}
```

Values supplied at runtime via `--params file.toml` or `--param beta=0.3`.
Parameters are the model's degrees of freedom — what you sweep, fit, or hold
fixed across analyses.

**Document them with `#'`.** A name and kind alone (`beta : rate`) rarely says
what a parameter _means_. A `#'` doc comment on the line above attaches a short
description that shows up in `camdl inspect` and travels with the model:

```camdl
parameters {
  #' per-capita transmission rate (contact rate × per-contact prob)
  beta  : rate
  #' mean infectious period is 1/gamma
  gamma : rate
}
```

This is recommended, not required — an unannotated model compiles fine — and
it's a graded habit worth building: annotate **parameters** first (they're the
easiest to forget and carry the most hidden meaning), then the non-obvious
**compartments**, and more as you want a self-describing model. The `#'`
describes what a parameter _is_; its _value_ still lives in a `--params` TOML,
never in the model.

### Let bindings: compile-time inlining

```camdl
let N = S + I + R
let foi = beta * I / N
```

Let bindings are **inlined at compile time.** Everywhere the compiler sees `N`
in a rate expression, it substitutes the expression `S + I + R` directly. The IR
contains no reference to `N` — just `PopSum(["S", "I", "R"])` in the expression
tree. Let bindings don't exist at runtime.

This means they're not "variables" in the programming sense — they're named
expression fragments. The substituted expression evaluates fresh at every use
(every Gillespie event, every timestep), using the current compartment values.
So `N` always reflects the current total population, even though it's "just" a
textual substitution.

### Tables: fixed data arrays

```camdl
tables {
  C_age    : age × age  = [[12.0, 4.0], [4.0, 8.0]]
  pop      : patch      = read("data/lga_pop.tsv")
  adj      : patch × patch = read("data/lga_adj.tsv", default = 0.0)
  coverage : patch      = read("data/lga_coverage.tsv")
}
```

Tables hold fixed data that doesn't change during simulation: contact matrices,
population sizes, spatial adjacency weights, vaccination coverage estimates,
distance matrices, historical immunization records. Anything that's
observational or structural input to the model — not a quantity you'd infer.

**Table dimensions come from the `dimensions {}` block.** The `age × age` in
`C_age : age × age` refers to the `age` dimension declared in
`dimensions { age
= [child, adult] }`. The compiler uses the dimension levels to
compute table sizes and resolve indexed lookups like `C_age[a, b]` into linear
indices.

**Table lookups are resolved at compile time.** `C_age[child, adult]` becomes
`TableLookup("C_age", Const(1))` in the IR — the index is a pre-computed
integer. At runtime, it's a single array access.

**The rule:** If it changes during simulation, use `let`. If it's supplied
externally and might be inferred or swept, it's a `parameter`. If it's fixed
input data — spatial structure, contact patterns, demographic data, coverage
estimates, historical records — it's a `table`.

---

## Dimensions: `[i in dim]`

Stratification adds dimensions to compartments. The index notation is designed
to mirror mathematical subscripts:

| Math notation           | camdl                                  | Meaning                                                         |
| ----------------------- | -------------------------------------- | --------------------------------------------------------------- |
| S_a ∀a ∈ {child, adult} | `S[a in age]`                          | for each age group                                              |
| S_child                 | `S[child]`                             | one specific stratum                                            |
| Σ_a S_a                 | `S`                                    | bare name = whole compartment (unindexed) = sum over all strata |
| N_a = S_a + I_a + R_a   | `let N[a in age] = S[a] + I[a] + R[a]` | per-stratum derived quantity                                    |

Declare a dimension:

```camdl
dimensions { age = [child, adult] }
stratify(by = age)
```

This expands every compartment: `S` becomes `S_child` and `S_adult` in the IR.
Transitions, let bindings, interventions, and init all use the same `[i in dim]`
syntax to iterate over dimension values.

Add age structure to the SIR:

```camdl
compartments { S, I, R }
dimensions { age = [child, adult] }
stratify(by = age)

parameters {
  beta  : rate
  gamma : rate
}

let N[a in age] = S[a] + I[a] + R[a]

tables {
  C : age × age = [[12.0, 4.0],
                    [4.0,  8.0]]
}

transitions {
  infection[a in age] : S[a] --> I[a]
    @ beta * S[a] * sum(b in age, C[a, b] * I[b] / N[b])

  recovery[a in age] : I[a] --> R[a]
    @ gamma * I[a]
}
```

The compiler expands to 4 transitions: `infection_child`, `infection_adult`,
`recovery_child`, `recovery_adult`. Each has a fully resolved rate expression
with concrete compartment names and numeric table indices.

---

## Multiple dimensions compose

Stack dimension declarations and `stratify` for a Cartesian product:

```camdl
dimensions {
  age   = [child, adult]
  patch = [north, south]
}
stratify(by = age)
stratify(by = patch)
```

This gives `S_child_north`, `S_child_south`, `S_adult_north`, `S_adult_south`.
Transitions bind one or both:

```camdl
# Within each patch, age-structured transmission
infection[a in age, p in patch] : S[a, p] --> I[a, p]
  @ beta * S[a, p] * sum(b in age, C[a, b] * I[b, p] / N[b, p])

# Between patches, same age group
importation[a in age, p in patch, q in patch] : S[a, p] --> I[a, p]
  @ kappa * adj[p, q] * S[a, p] * I[a, q] / N[a, q]
  where p != q
```

The `where p != q` guard filters self-loops at compile time — patches don't
import from themselves.

---

## Interventions: scheduled discrete events

Transitions run continuously. Interventions fire at specific times and move
individuals instantaneously.

```camdl
interventions {
  sia : transfer(fraction = 0.8, from = S, to = V) at [180, 545]
}
```

At day 180 and 545, 80% of S moves to V. Times are in the model's `time_unit`.

Interventions support `[i in dim]`, table lookups in expressions, and the full
composability of transitions:

```camdl
parameters {
  vacc_eff : probability   # vaccine effectiveness (tunable)
}

tables {
  coverage : patch = read("data/lga_coverage.tsv")  # per-LGA field data (fixed)
}

interventions {
  sia[p in patch] : transfer(
    fraction = vacc_eff * coverage[p],
    from = S[p], to = V[p]
  ) at [180, 545]
}
```

One declaration, 238 expanded interventions — each with its own coverage from
observational data. `vacc_eff` (tunable) × `coverage[p]` (data) separates what
you infer from what you measured.

**With age stratification, target a specific group:**

```camdl
# SIA vaccinates under-5 only
sia[p in patch] : transfer(
  fraction = vacc_eff * coverage[p],
  from = S[under5, p], to = V[under5, p]
) at [180, 545]
```

Bound `p` iterates patches. Concrete `under5` is fixed. Adults are untouched.

---

## Scenarios: counterfactual comparisons

Interventions are **disabled by default** — the baseline is natural transmission
only. Scenarios select which interventions fire and optionally modify
parameters.

```camdl
scenarios {
  baseline {
    label = "no vaccination"
  }
  with_sia {
    enable = [sia]
  }
  high_transmission {
    scale = { beta = 1.5 }
  }
  combined {
    compose = [with_sia, high_transmission]
  }
}
```

`enable = [sia]` resolves the family name `sia` to all 238 expanded
interventions. This is name resolution, not string matching — the compiler tags
each expanded intervention with its family name.

| Keyword                      | Effect                                           |
| ---------------------------- | ------------------------------------------------ |
| `enable = [names]`           | Activate interventions (families or individuals) |
| `disable = [names]`          | Deactivate interventions                         |
| `set = { param = value }`    | Override parameter values                        |
| `scale = { param = factor }` | Multiply parameter values                        |
| `compose = [A, B]`           | Apply A then B in sequence                       |

Select a scenario at the CLI:

```bash
camdl simulate model.camdl --params p.toml --scenario with_sia --seed 42
```

Or run all scenarios across many seeds:

```bash
camdl batch run experiment.toml --parallel 8
```

---

## Init: starting conditions

The `init` block sets compartment values at t=0. Supports `[i in dim]` with
override-by-source-order:

```camdl
init {
  S[p in patch] = N0[p]                         # bulk-set all patches
  S[borno_damboa] = N0_borno_damboa - I0        # override seed patch
  I[borno_damboa] = I0
}
```

Later entries overwrite earlier ones for the same compartment. Pattern: bulk-set
everything, then patch the exceptions.

---

## Observations: connecting simulation to data

```camdl
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
```

`incidence(infection)` counts infection events since the last observation time.
The `neg_binomial` likelihood defines how simulated counts relate to observed
data — used for scoring during inference and for generating synthetic
observations during forward simulation.

---

## Putting it together

A 238-patch Nigerian polio model in ~55 lines:

```camdl
time_unit = 'days

compartments { S, E, I, R, V }

dimensions {
  patch = read("data/lga_pop.tsv", column = "patch")   # 238 LGAs from data
}

stratify(by = patch)

let N[p in patch] = S[p] + E[p] + I[p] + R[p] + V[p]

parameters {
  sigma    : rate        in [0.001, 1.0]
  gamma    : rate        in [0.001, 1.0]
  omega    : rate        in [0.0001, 0.1]
  kappa    : rate        in [0.0, 0.5]
  vacc_eff : probability in [0.0, 1.0]
  I0       : count       in [1, 10000]
  R0[patch] : positive   in [0.5, 15.0]
  N0[patch] : count      in [100, 10000000]
}

let beta[p in patch] = R0[p] * gamma

tables {
  adj      : patch × patch = read("data/lga_adj.tsv", default = 0.0)
  coverage : patch         = read("data/lga_coverage.tsv")
}

transitions {
  infection[p in patch] : S[p] --> E[p]
    @ beta[p] * S[p] * I[p] / N[p]

  importation[p in patch, q in patch] : S[p] --> E[p]
    @ kappa * adj[p, q] * S[p] * I[q] / N[q]
    where p != q

  progression[p in patch] : E[p] --> I[p]  @ sigma * E[p]
  recovery[p in patch]    : I[p] --> R[p]  @ gamma * I[p]
  waning[p in patch]      : V[p] --> S[p]  @ omega * V[p]
}

interventions {
  sia[p in patch] : transfer(
    fraction = vacc_eff * coverage[p],
    from = S[p], to = V[p]
  ) at [180, 545]
}

init {
  S[p in patch] = N0[p]
  I[borno_damboa] = I0
}

simulate {
  from = 0 'days
  to   = 730 'days
}

scenarios {
  baseline { label = "no SIA" }
  with_sia {
    label = "with SIA"
    enable = [sia]
  }
}
```

---

## Quick reference

| Concern            | Syntax                                          | Notes                             |
| ------------------ | ----------------------------------------------- | --------------------------------- |
| Compartments       | `compartments { S, I, R }`                      | what exists                       |
| Dimensions         | `dimensions { age = [...] }` + `stratify(by=X)` | Cartesian product                 |
| Derived quantities | `let N = S + I + R`                             | inlined at compile time           |
| Parameters         | `parameters { beta : rate }`                    | external, supplied at runtime     |
| Tables             | `tables { C : age × age = [...] }`              | fixed data arrays                 |
| Continuous flow    | `src --> dst @ propensity`                      | Gillespie / chain-binomial events |
| Scheduled events   | `transfer(...) at [times]`                      | interventions, discrete           |
| Observations       | `incidence(...)`, likelihood                    | inference + synthetic data        |
| Initial state      | `init { S = expr }`                             | override-by-source-order          |
| Scenarios          | `enable`, `set`, `scale`, `compose`             | counterfactual selection          |
| Time range         | `simulate { from, to }`                         | defaults, overridable             |

---

## Known limitations

**Index-only dimensions require a `dimensions {}` entry.** You can declare
`dimensions { round = [r1, r2, r3] }` without a `stratify` to create a dimension
that indexes tables but doesn't expand compartments. This supports per-patch
scheduling tables like `sia_day : patch × round`. Table-only dimensions are
declared the same way as stratification dimensions.

**Observation block not yet exercised in golden tests.** The syntax compiles but
end-to-end inference (scoring, particle filter) is not yet implemented.
Observations currently generate synthetic data during forward simulation.

**String literals in scenarios are fragile.** `label = "multi word"` may not
parse correctly for all token patterns. Labels are cosmetic and don't affect
simulation.
