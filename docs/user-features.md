# camdl User Features

What makes camdl pleasant to write models in.

---

## Write the math, not the code

camdl reads like the math it represents. A transition is "from → to at rate." An
index is a mathematical subscript. A table is a lookup array.

```camdl
infection[a in age] : S[a] --> I[a]
  @ beta * S[a] * sum(b in age, C[a, b] * I[b] / N[b])
```

No hidden multiplication by population counts. No implicit scope rules. The rate
is the total propensity — what you'd write on paper.

---

## Physical units

Unit literals prevent the most common class of modeling errors: rate/duration
confusion. The compiler tracks dimensions and converts at compile time.

```camdl
time_unit = 'days

parameters {
  gamma : rate        # 1/time
  mu    : rate
}

tables {
  age_dur : age 'years = [5, 60]           # durations in years
  mu_age  : age 'per_day = [0.00007, 0.00004]  # rates in per-day
}

simulate {
  from = 0 'days
  to   = 5 'years      # automatically converted: 5 × 365.25 = 1826.25 days
}
```

Supported: `'days`, `'weeks`, `'months`, `'years`, `'per_day`, `'per_week`,
`'per_month`, `'per_year`. Mixed-unit arithmetic works:
`0.1 'per_day * 5 'days = 0.5` (dimensionless). Adding a rate to a duration is a
compile error.

---

## Anchored vs unanchored, and calendar stepping

A model is **anchored** if it declares `origin = date(...)`; otherwise
**unanchored**. The split controls how dates and calendar arithmetic behave —
see [`docs/dates.md`](dates.md) for the full reference.

```camdl
# (a) Anchored: daily axis, per-month rate parameters
time_unit = 'days
origin    = date("2020-02-24")

parameters {
  beta  : positive 'per_month     # compiler converts to per-day at compile time
  gamma : positive 'per_month
  tau   : instant in [date("2020-01-21"), date("2020-04-30")]
                                   # renders as a date in fit summary
}
```

```camdl
# (b) Unanchored: monthly axis (the dacca SIRS shape)
time_unit = 'months
# no origin — t = 0 has no calendar meaning

parameters {
  beta  : positive 'per_month
  gamma : positive 'per_month
}

let latent = 1 'months              # affine 30.44 days as a length
```

```camdl
# (c) Calendar stepping (anchored mode only)
let mid_year       = add_calendar_months(origin, 6)
let school_quarter = date_range(origin, date("2025-01-01"), calendar_months = 3)
                                   # 21 quarterly breakpoints, calendar-aligned

simulate { from = origin to = add_calendar_years(origin, 5) }
```

`add_calendar_months` / `add_calendar_years` are the _only_ way to step a date
by calendar months/years; month-end clamping is canonical
(`date("2021-01-31") + 1 month = date("2021-02-28")`). Direct
`date(...) + N 'months` is a hard error (**E321**) — the language forbids the
silent affine-drift that would produce.

In an anchored model, write absolute-time positions as `date(...)` rather than
as bare numbers — not only `simulate { from / to }` but the `at = [...]`
schedule of any `interventions {}` or `events {}` entry. Each `date(...)`
resolves to its internal offset through `origin`, so the model reads in calendar
terms instead of opaque day counts:

```camdl
# `from = 730` says "730 days after origin"; the reader has to do the arithmetic.
simulate { from = date("1952-01-01")  to = date("1963-09-08") }

events {
  importation : add(I, 1) at [date("1952-03-15"), date("1955-09-01")]
}
```

A bare number in one of these positions under `origin = date(...)` is read as
internal-time units from origin and warns — **W324** in `simulate.from/to`,
**W325** in an `at`-schedule — pointing you to the `date(...)` form (or an
explicit `<n> 'days` if the offset really is intentional). This mirrors the data
loader's **W326** nudge on numeric `--data` time columns, so the calendar-vs-raw
choice is surfaced the same way on both the model and the data side.

---

## Calendar-based forcing with range syntax

Specify school terms, work weeks, or campaign windows as day ranges instead of
raw arrays. The compiler generates the values.

```camdl
forcing {
  # UK school calendar (He et al. 2010)
  school : periodic 'ratio {
    period = 365.25 'days
    step   = 1 'days
    on     = [7:100, 115:199, 252:300, 308:356]
  }
}
```

Four ranges, one line. The compiler produces 365 bins with exactly 277 school
days (fraction = 0.7589). If you use `step = 7 'days` with day-granularity
ranges, the compiler warns that endpoints don't align to the step size (W301).

Use `school(t)` in rate expressions — the `(t)` makes the time dependency
explicit. Bare `school` also works.

---

## Stochastic process control

Rate wrappers control how event counts are drawn per transition. The default is
Poisson (demographic stochasticity). Two alternatives for specific modeling
needs:

```camdl
transitions {
  # Standard: count ~ Poisson(rate × dt)
  recovery : I --> R  @ gamma * I

  # Extra-demographic noise: count ~ NegBinomial (He et al. 2010)
  # Gamma noise on the rate — variance scales quadratically with mean
  infection : S --> E  @ overdispersed(beta * S * I / N, sigma_se)

  # Deterministic rounding: count = nearbyint(rate × dt)
  # For demographic flows where Poisson noise is unphysical
  birth : --> S  @ deterministic((1.0 - cohort) * daily_births)
}
```

Models with `overdispersed()` transitions produce a hard error on
`--backend gillespie` — the capabilities system catches incompatible backend
choices before simulation starts.

---

## Math functions and time

`t` is the current simulation time, available anywhere in expressions. Standard
math functions work as expected:

```camdl
let day_of_year = mod(t, 365.25)
let pop_decay = N0 * exp(-mu * t)
let threshold = if I > floor(sqrt(N)) then 1.0 else 0.0
```

Available: `exp`, `log`, `sqrt`, `abs`, `floor`, `ceil`, `mod`.

---

## Named indexing

When a compartment has multiple dimensions, use named indices to avoid
positional ambiguity:

```camdl
dimensions {
  age   = [child, adult]
  patch = [north, south, east]
}

# Positional: first index = age, second = patch
S[child, north]

# Named: order doesn't matter, intent is clear
S[patch = north, age = child]

# To marginalize a dimension, sum over it explicitly — there is no
# implicit summation. A partial index (some dimensions given, others
# omitted) is an error (E287); spell the sum out:
sum(p in patch, S[age = child, patch = p])
#   = S[child, north] + S[child, south] + S[child, east]
```

---

## Data-driven dimensions

Dimension levels can come from data files. No manual enumeration of 774 district
names:

```camdl
dimensions {
  patch = read("data/population.tsv", column = "district")
}

tables {
  pop : patch = read("data/population.tsv")
  adj : patch × patch = read("data/adjacency.tsv", default = 0.0)
}
```

The compiler validates every table entry against the known dimension levels.
Typos produce an error with a Levenshtein suggestion.

---

## Iteration primitives

Three composable patterns cover all structured transitions:

```camdl
# For each value in a dimension
infection[a in age] : S[a] --> I[a]  @ beta * S[a] * I[a] / N[a]

# For consecutive pairs (aging, Erlang sub-stages)
aging[(a, a_next) in consecutive(age)] : S[a] --> S[a_next]
  @ (1 / age_dur[a]) * S[a]

# For every integer compartment (death, migration)
death[c in compartments, a in age] : c[a] -->  @ mu * c[a]
```

Combine with `where` guards for compile-time filtering:

```camdl
migration[c in compartments, src in patch, dst in patch]
  : c[src] --> c[dst]
  @ theta * pop[dst] / (distance[src, dst] ^ 2) * c[src]
  where src != dst
```

The compiler expands the Cartesian product and filters at compile time. 774² =
599,076 candidate transitions, minus 774 self-loops, in one declaration.

---

## Scenarios as counterfactual patches

Interventions are off by default. Scenarios select which fire:

```camdl
scenarios {
  baseline {
    label = "no SIA"
  }
  with_sia {
    enable = [sia]
    set = { vacc_eff = 0.95 }
  }
  high_transmission {
    scale = { beta = 1.5 }
  }
  combined {
    compose = [with_sia, high_transmission]
  }
}
```

CRN coupling: same seed with different scenarios produces correlated
trajectories. Pre-intervention trajectories are byte-identical.

---

## Reactive interventions (state-triggered policy)

A fixed `at [...]` schedule says _when_ a campaign happens. A **reactive
intervention** says _what triggers it_ — the campaign fires as a function of
what surveillance has detected, which is how real outbreak response works ("run
an SIA after AFP detections cross a threshold") and the only way native EVSI —
the value of expanded surveillance — is meaningful, since under a fixed schedule
extra surveillance changes nothing.

```camdl
reactive_interventions {
  mop_up : when sum_observed(weekly_afp, window = 28 'days) >= afp_threshold {
    after    = 21 'days       # respond 3 weeks after the trigger
    action   = transfer(fraction = sia_coverage, from = S, to = V)
    once     = false
    cooldown = 180 'days      # don't re-fire for 6 months
  }
}
```

The `when` predicate reads **observed data**, not latent truth:
`observed(stream)` is the latest reported value and
`sum_observed(stream, window = D)` the trailing sum — the distinction matters
because a health ministry acts on reported cases, not on the model's hidden
infection count. The reported value is the _realized_ random draw from the
observation model (e.g. a Poisson report count), the same number a `--obs` file
would contain — not its expectation — so the trigger behaves exactly as
surveillance would. Like `interventions {}`, a reactive policy is
scenario-toggleable, so a `with_response` scenario `enable`s it and a `baseline`
omits it.

Forward simulation on the **chain-binomial** backend executes the agenda: the
policy fires when its trigger crosses, `after` the lag elapses, honouring
`once`/`cooldown`. Every firing is recorded in the run's `reactive_log.tsv`
artifact (`trigger_time`, `policy`, `trigger_value`, `threshold`, `fire_time`,
`action`); `camdl cat <id> --stream reactive_log.tsv` reads it, and
`--reactive-log PATH` mirrors it. Inference (IF2/PGAS/PMMH) and the
Gillespie/ODE forward backends do not yet run reactive policies — an active
reactive policy there stops with a clear capability error rather than silently
ignoring the policy. See the spec (`camdl docs language`, §13.9) for the full
surface.

---

## Reporting derived quantities

A `quantities {}` block reports summaries of a run without pretending they are
data — the non-scored twin of an observation. You no longer have to smuggle a
peak or an attack rate through a fake scored stream.

```
quantities {
  prevalence      = I / N                       # a series (one value per output time)
  attack_rate     = final((N0 - S) / N0)        # a scalar
  peak_prevalence = max(I / N)
  time_to_peak    = time_of_max(I)              # a time — rendered as a date in an anchored model
  takeoff         = first_above(I_total, i_thr) # the first time I_total exceeds i_thr
  fadeout         = last_above(I_total, 0)      # the last time I_total is above 0
  outbreak_dur    = fadeout - takeoff           # arithmetic over reduced scalars
}
```

A state quantity with no reduction is a **series** — one value per output time.
This is how you emit a _derived channel_ the model computes but doesn't carry as
a compartment: force of infection `λ(t)`, effective reproduction number,
cumulative incidence, EIR, prevalence `I / N`. Declaring it here — rather than
reconstructing it in a downstream script from `traj.tsv` — keeps it in step with
the model's own arithmetic and bands it over the posterior in `fit predict`. A
reduction (`final`, `max`, `mean`, `time_of_max`, `first_above`, `integral`,
`count_above`, …) instead collapses the series to a **scalar**. A reduction can
also fold a **simulated observation** —
`peak_reported = max(observations.cases)` reduces the same `y_sim` the run drew
for the declared `cases` stream (never a fresh draw); an observation source must
be reduced (a bare `observations.cases` series is rejected). They run wherever a
simulation does — over prior-predictive draws (`simulate --draws`), over a
fitted posterior (`fit predict`), banded into `quantities/<name>.tsv` with a
`quantities.json` manifest. A timing question that never resolves (an outbreak
that never takes off in a given draw) is reported as **right-censored**, not as
a fabricated time. Because they are derived reports, adding or changing a
`quantities {}` block never re-keys a model's `run_id`. The same holds for `#'`
documentation: it lands in the IR envelope's `docs` dictionary, outside the
`model` object identity is computed from, so correcting a citation costs you no
fits.

---

## Inspect without simulating

`camdl dev eval` evaluates time-dependent expressions at a grid without running
a simulation. Useful for verifying forcing curves, covariates, and parameter
formulas:

```bash
camdl dev eval model.camdl --params p.toml --expr "school" --from 0 --to 365 --every 1
```

`dev eval` resolves forcing functions and parameters; `let`-bindings (such as a
seasonal `seas` term) are not directly evaluable. Output is TSV — pipe to a
file, load in polars/R, plot. If an expression references compartment state, the
error message directs you to run `camdl simulate <model> -o traj.tsv` instead
(which writes compartment and `flow_*` columns per step).

---

## Particle filter diagnostics

`camdl pfilter --trace` shows one-step-ahead predictions alongside the data, not
just a log-likelihood number:

```
time  ll_increment  ESS    obs_mean  obs_q05  obs_q50  obs_q95  state_mean  state_q05  state_q50  state_q95  observed
7     -7.84         17.4   42.3      5        31       112      84.1        11         63         220        82
14    -5.37         217.7  51.2      12       45       98       103.4       25         91         197        98
```

See exactly where the model predicts well (data inside the 90% interval) and
where it fails.

---

## Gradient-based ODE fitting

The ODE backend is a fitting backend, not only a forward simulator. Integrating
the mean-field skeleton gives one trajectory per parameter point, so the
deterministic likelihood `p(y | θ, ODE skeleton)` evaluates directly — no
particle filter — and `camdl fit` samples the posterior on it two ways:

- **`mh`** — gradient-free adaptive Metropolis-Hastings. Robust default; runs on
  any ODE model the backend can integrate.
- **`nuts`** — the No-U-Turn Sampler driven by **symbolic forward
  sensitivities**. The compiler differentiates the rate and observation
  expressions source-to-source and carries `∂x/∂θ` through fixed-step `rk4`, so
  the gradient is exact — no finite differences. NUTS moves through the
  correlated, ridge-shaped posteriors that stall gradient-free MH.

```toml
[stages.posterior]
algorithm = "nuts" # or "mh" for the gradient-free sampler
backend = "ode"
chains = 4
warmup = 500
samples = 500
```

`nuts` needs a **differentiable model**: the capability gate refuses — naming
the reason — an undifferentiable rate/observation gradient, an adaptive `rk45`
integrator, a scheduled `interventions {}` / `events {}` effect, or a
parameterized (`ivp`) initial condition; fit those with gradient-free `mh` or
the stochastic-process methods instead. This targets a different statistical
object than the stochastic backends — `p(y | θ, ODE skeleton)` rather than
`p(y | θ)` — so see `camdl docs inference` (the ODE-backend fitting section) for
when to pick which. Both samplers are `[beta]`.

---

## Compiler diagnostics

The compiler catches errors at compile time with domain-specific messages:

```
error[E100]: parameter name 't' is reserved for simulation time
  = hint: choose a different name

error[E332]: 'sex' is not a dimension of table 'C_age'
  = hint: its dimensions are: age, age

warning[W301]: periodic range 7:100 is not aligned to step size 7
  = hint: use step = 1 for exact boundaries
```

Dimension mismatches, missing indices, wrong function arities, reserved name
collisions, and unit errors are all caught before simulation starts.

---

## Content-addressable output

Every simulation run is stored in a directory determined by its inputs:

```
runs/{sim_hash}/{scenario}-{scen_hash}/seed_{N}/
```

Same inputs → same hash → cached. Different inputs → different directory. Add
more seeds without re-running existing ones. Change one scenario without
invalidating others.

---

## Multiple simulation backends

One model, three backends. Choose the right tradeoff:

| Backend          | When to use                                       |
| ---------------- | ------------------------------------------------- |
| `gillespie`      | Small populations, extinction matters             |
| `chain_binomial` | Euler-multinomial (matches pomp's reulermultinom) |
| `ode`            | Deterministic parameter sweeps                    |

```bash
camdl simulate model.camdl --params p.toml --backend chain_binomial --dt 0.5 --seed 42
```

The chain-binomial uses true multinomial competing-risk draws with deferred
state updates — the exact Euler-multinomial algorithm used in the pomp
ecosystem.

---

## Why camdl: a side-by-side comparison

The He et al. (2010) London measles model — the same model, in pomp and camdl.

### School-term forcing

**pomp** — 20 lines of C inside a string:

```c
// Inside Csnippet("...")
seas = 1.0 - amplitude;
if ((t-floor(t)) >= 7.0/365.0 && (t-floor(t)) <= 100.0/365.0)
  seas = 1.0 + amplitude * 0.2411/0.7589;
else if ((t-floor(t)) >= 115.0/365.0 && (t-floor(t)) <= 199.0/365.0)
  seas = 1.0 + amplitude * 0.2411/0.7589;
else if ((t-floor(t)) >= 252.0/365.0 && (t-floor(t)) <= 300.0/365.0)
  seas = 1.0 + amplitude * 0.2411/0.7589;
else if ((t-floor(t)) >= 308.0/365.0 && (t-floor(t)) <= 356.0/365.0)
  seas = 1.0 + amplitude * 0.2411/0.7589;
```

**camdl** — 4 ranges:

```camdl
forcing {
  school : periodic 'ratio {
    period = 365.25 'days
    step   = 1 'days
    on     = [7:100, 115:199, 252:300, 308:356]
  }
}
let seas = 1.0 - amplitude + amplitude * (1.0 + 0.2411 / 0.7589) * school(t)
```

### Transmission with overdispersion

**pomp** — manual Gamma draw and rate arithmetic:

```c
dw = rgammawn(sigmaSE, dt);
beta = R0 * (gamma+mu) * seas;
foi = beta * pow(I+iota, alpha) / pop * dw/dt;
rate[0] = foi;
rate[1] = mu;
reulermultinom(2, S, &rate[0], dt, &trans[0]);
S += nearbyint(pop*br*dt) - trans[0] - trans[1];
```

**camdl** — the transition reads as math:

```camdl
infection : S --> E  @ overdispersed(beta * seas * S * ((I + iota) ^ alpha) / pop(t), sigma_se)
```

The `overdispersed()` wrapper handles the Gamma-Poisson compound internally. The
compiler expands the stoichiometry. The runtime handles competing risks. No
manual index arithmetic.

### Observation model

**pomp** — 8 lines of C:

```c
double m = rho*C;
double v = m*(1.0-rho+psi*psi*m);
double tol = 1e-18;
if (cases > 0.0)
  lik = pnorm(cases+0.5,m,sqrt(v),1,0) - pnorm(cases-0.5,m,sqrt(v),1,0) + tol;
else
  lik = pnorm(0.5,m,sqrt(v),1,0) + tol;
if (give_log) lik = log(lik);
```

**camdl** — one block:

```camdl
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(recovery)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
```

Observation-model parameters (`rho`, `k` here, and parameters inside a derived
projection such as `qgam * prevalence`) are estimated by gradient-based NUTS on
the same footing as transition-rate and overdispersion parameters: the compiler
differentiates the rate, likelihood, and σ² terms analytically, so there is no
finite-difference approximation and no silent zero. The set of differentiable
positions is derived from the types rather than maintained by hand, so a newly
added argument is covered by construction — it cannot be silently missed.

Where a parameter reaches the model through something with a live value but no
emitted gradient — a periodic step-value coefficient, a forcing's time-shift
(`lag`), an inline-table value chosen by a non-constant index, or a coefficient
reached _only_ through an initial condition (camdl computes no gradient for
initial-condition expressions) — NUTS is refused with a message naming the
parameter and the reason, rather than proceeding on a zero, silently biased
gradient; a binomial denominator `n`, which must be θ-independent, is refused
the same way. A genuinely structural coefficient — a spline, interpolation, or
piecewise knot fixed at construction — is a compile-time error instead, since it
cannot be estimated by any method. Gradient-free methods (IF2, particle-filter
PMMH) estimate the live-but-undifferentiated cases unchanged; only NUTS needs
the gradient.

### Parameter transforms

**pomp** — separate declaration, manual enumeration:

```r
partrans = parameter_trans(
  log = c("R0","sigma","gamma","alpha","iota","sigmaSE","psi"),
  logit = c("rho","cohort","amplitude"),
  barycentric = c("S_0","E_0","I_0","R_0")
)
```

**camdl** — derived from parameter types:

```camdl
parameters {
  R0        : positive       # → log transform
  sigma     : rate           # → log transform
  rho       : probability    # → logit transform
  amplitude : probability    # → logit transform
}
```

No separate declaration. The type system implies the transform. If you declare a
parameter as `probability`, the inference engine knows it lives on [0,1] and
uses logit. You can't accidentally forget to list a parameter in the transform
declaration.

### The model as a whole

pomp stitches together C code strings, R function calls, covariate tables,
parameter name vectors, and state variable lists. The model structure
(compartments, transitions, stoichiometry) is implicit in the C snippets — you
have to read the code to know that `trans[0]` is infection and `rate[2]` is
sigma.

camdl is one file where every piece has a name: compartments are declared,
transitions read as "from → to at rate," tables have typed dimensions, and the
compiler validates everything at compile time. A dimension mismatch, a missing
index, or a unit confusion produces a clear error before simulation starts — not
a segfault in dynamically compiled C code at runtime.
