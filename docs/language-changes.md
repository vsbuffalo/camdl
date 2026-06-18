# CAMDL language changes

Breaking and notable changes to the **CAMDL language** — grammar, dimensions,
and semantics — newest first. This is the history an agent needs when a model
that "should" compile is rejected: find the change, apply the migration.

Scope: the _language surface_ (what you write in a `.camdl` file). CLI and
`fit.toml` changes live in the full changelog (`camdl docs changelog`). For the
_current_ syntax, see `camdl docs language` (the spec).

How to read an entry: **what changed**, the **migration** (old → new), and the
**diagnostic** you'll hit if you use the old form.

---

## 2026-06-18 — `reactive_interventions {}` block + new reserved words (gh#204)

**What.** A new top-level block, `reactive_interventions {}`, declares
state/observation-triggered policies whose timing the model discovers at run
time (e.g. "run an SIA after AFP detections cross a threshold"):

```
reactive_interventions {
  mop_up : when sum_observed(weekly_afp, window = 28 'days) >= threshold {
    after    = 21 'days
    action   = transfer(fraction = cov, from = S, to = V)
    once     = false
    cooldown = 180 'days
    scope    = exogenous
  }
}
```

The `when` predicate is a boolean over the trigger inputs `observed(stream)` and
`sum_observed(stream, window = D)`, combined with `and` / `or` / `not`. Reactive
policies lower to the IR's new `fire = Reactive(..)` source — internally the
intervention `schedule` field became `fire: Scheduled | Reactive`, an orthogonal
axis from `kind` (a reactive policy is `kind = Scenario, fire = Reactive`). They
are parsed, dimension-checked, and emitted to the IR, but no simulation backend
executes the reactive agenda yet. (IR schema 0.16 → 0.17.)

**Migration.** Three new reserved words — `reactive_interventions`, `when`,
`action` — are added. A model that used one as a compartment / parameter /
dimension name must rename it. `observed(...)` / `sum_observed(...)` are
recognized only inside a `when` predicate. No other change to existing models
(the `fire` IR reshape is internal; existing `.camdl` source is unaffected).

**Diagnostic.** `when` / `action` / `reactive_interventions` as an identifier →
**E001**. `observed()` / `sum_observed()` in a rate or other model expression →
**E278** ("only valid inside a reactive trigger predicate"). Reactive
validation: once+cooldown contradiction **E276**; negative `after` / `cooldown`
**E274**; non-comparison `when` **E273**; threshold not a constant/parameter
**E272**; stream-arg / window arity **E270** / **E271**; bad `scope` **E277**.
Running a model with an _active_ reactive policy → a `REACTIVE_INTERVENTIONS`
capability error (parsed but not yet executable).

## 2026-06-16 — `log_uniform` and `truncated_normal` prior distributions (gh#155)

**What.** Two new priors for the `~ dist(...)` syntax:

- `~ log_uniform(lower, upper)` — uniform on the log scale (every order of
  magnitude equally likely); the honest weakly-informative choice for a scale
  parameter known only to within orders of magnitude, where `log_normal`
  overstates knowledge. Requires the parameter's `Log` transform.
- `~ truncated_normal(mean, sd)` — a normal truncated to the parameter's
  declared `in [lo, hi]` bounds, which are the **single source of truth** for
  the support (exact inverse-CDF sampling, so no draw-time rejection unlike
  `normal(...)` + `in [..]`). The parameter MUST declare `in [lo, hi]`.

Both reject hierarchical/pooled use. Purely additive — no existing model
changes. (IR schema 0.15 → 0.16.)

**Migration.** None required (additive).

**Diagnostic.** `truncated_normal` without a declared `in [lo, hi]` → **E285**;
either new prior used as a hierarchical/pooled prior → **E286**.

## 2026-06-16 — `positive` / `real` accept an optional unit literal (gh#60)

**What.** The dimension-under-determined parameter kinds `positive` and `real`
now accept an optional **tier-3 unit literal** that supplies the parameter's
dimension, in the same grammar slot a `[dim]` bracket annotation would use:

```camdl
parameters {
  tau    : positive 'ratio    in [0.001, 3.0]    # dimensionless (= positive [1])
  iota   : positive 'count                        # a count       (= positive [P])
  importn: positive 'per_year                     # per-time rate (= positive [T^-1])
}
```

Only the unit's **dimension** is read — a parameter's value is always in the
model `time_unit`, so the scale half is inert (`'per_year` and `'per_day` set
the same T⁻¹ dimension). The literal is exact sugar for the matching bracket
annotation. This **resolves the I300** ("dimension of parameter could not be
determined") a bare `positive`/`real` emits in a dimension-determined slot, and
makes a dimensional misuse of the parameter a hard error rather than a swallowed
info. This is **purely additive** — every existing `positive`/`real` declaration
is unchanged. No IR schema change (the literal lowers into the existing
`param_dim` field).

**Migration.** None required. To type a previously-undetermined `positive`:
replace `tau : positive` with `tau : positive 'ratio` (or the appropriate unit).

**Diagnostic.** A unit literal on a kind whose dimension is already fixed
(`rate`, `probability`, `count`, `instant`, `duration`) is **E281** ("a unit
literal is only allowed on the 'positive' and 'real' kinds"). A unit literal
combined with a `[dim]` bracket on the same declaration is **E282**.

## 2026-06-16 — `simulate {}` gains a tagged `integrator` (gh#166)

**What.** The `simulate {}` block gains an optional **tagged integrator**
selecting the ODE method and (for rk45) its adaptive tolerances:

- **`integrator = rk4`** — fixed-step classic RK4 (the default; omit
  `integrator` entirely for it). Unchanged behaviour.
- **`integrator = rk45 { atol = 1e-8  rtol = 1e-6 }`** — adaptive Dormand–Prince
  RK4(5) (opt-in; large steps in smooth stretches, small steps only where the
  trajectory changes fast). `atol`/`rtol` are **dimensionless** (tolerances, not
  times), optional (omitted → the runtime's calibrated default), and are **keys
  of the `rk45` block** — they cannot be written without rk45.

The tolerances live _inside_ the `rk45` tag by design: the IR type is
`Rk4 | Rk45 { atol, rtol }`, so an orphan tolerance (atol without rk45, or rk4
with tolerances) is unrepresentable. This is **purely additive** — every
existing model is unaffected (no `integrator` → `rk4`). The IR schema version
bumped **0.14 → 0.15**; old IR deserializes to `rk4`.

**Migration.** None required. To opt a model into adaptive stepping:

```camdl
simulate {
  from = 0 'years
  to   = 40 'years
  integrator = rk45 { atol = 1e-8  rtol = 1e-6 }   # or just `integrator = rk45`
}
```

**CLI.** `camdl simulate --integrator rk4|rk45` overrides the method for a
forward run (method-only; it mutates the model before the run-id is computed, so
the choice is recorded in the content hash, and it preserves any model-declared
tolerances). There is intentionally **no `fit --integrator`**: the integrator is
part of the model's content identity, so on the inference path it is set in the
model's `simulate {}` block, not on the command line.

**Diagnostics.**

- `integrator = rk99` → **E106**
  `unknown integrator 'rk99': expected rk4 or rk45`.
- `integrator = rk4 { atol = 1e-8 }` → **E106**
  `` `integrator = rk4` takes no
  tolerances `` (atol/rtol are rk45-only).
- `integrator = rk45 { foo = 1 }` → **E106** `unknown integrator option 'foo'`.
- `atol = 1e-8` at the top level (outside the `rk45` block) → **E106**
  `unknown
  simulate key 'atol'`.
- `atol = 1e-8 'days` (any unit) → **E106** `` `atol` must be dimensionless ``.
- A `dt` / `integrator` key inside a **scenario** `simulate {}` block → **E106**
  (whole-model knobs; set them once at the top level).
- `integrator = rk45` on a model that references `dt` in a rate (`Expr::Dt`) →
  rejected at simulation: adaptive stepping has no single fixed `dt`; use `rk4`.

---

## 2026-06-10 — observation block: `~` measurement, `columns {}`, `from`, `emit_schedule` (gh#171)

**What.** The `observations {}` surface was reshaped so it reads like the rest
of the language and binds data **by name, never positionally**:

- The measurement model is written with **`~`** (the operator already used for
  priors): `cases ~ neg_binomial(...)`, replacing
  `likelihood = neg_binomial(...)`. The left side is a declared value column;
  the right side is keyword-only.
- A **`columns { name : role }`** block (always required) declares every file
  column and its role — `time`, `dim`, or a value type (`count`/`real`/
  `probability`/…). The data file binds to these names; the `: time` column is
  the fit time source (no more "column 0 is time").
- The **stream-header colon is dropped**: `cases : { … }` → `cases { … }`, with
  an optional **`from <source>`** clause naming the data source a file binds to
  (`--data <source>=file`; defaults to the stream name), so several streams can
  read one wide file.
- The emission cadence is renamed **`emit_schedule`** and is **simulate-only**
  (it tells `simulate --obs` when to emit synthetic rows). It is **optional** —
  a fit-only model omits it; fitting reads the data file's `time` column.

**Migration** (old → new):

```camdl
# OLD
observations {
  cases : {
    projected  = incidence(infection)
    every      = 7 'days
    likelihood = neg_binomial(mean = rho * projected, r = k)
  }
}

# NEW
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days   # simulate-only; omit for a fit-only model
    cases         ~ neg_binomial(mean = rho * projected, r = k)
  }
}
```

The `projected = …` field is unchanged. The `~` RHS does **not** take the
prior's `| dim` pooling suffix (stratify the stream header instead).

**Diagnostic.** `error[E273]` `likelihood = D(...)` → `<col> ~ D(...)`;
`error[E270]` stream-header colon removed → `name { … }`; `error[E272]`
`every`/`schedule` → `emit_schedule`; `error[E271]` the `| dim` suffix is a
prior construct, not a likelihood one. New-surface coherence: `E274` unknown
column role, `E275` missing/duplicate `: time`, `E276` undeclared scored column,
`E277` dead value column, `E278` `[p in dim]` ↔ `: dim` mismatch.

## 2026-06-09 — forcing/table coefficients are live parameters (gh#119)

**What.** A parameter used inside a forcing coefficient (`amplitude = alpha`) or
an inline-table value (`tbl = [k, ...]`) is now evaluated **live** during
inference, so it is genuinely estimable — previously it was frozen at its
construction-time value (a silent flat likelihood; see the incident
`2026-06-09-forcing-coefficient-param-frozen-at-construction.md`). Sinusoidal
and Fourier coefficients, and constant-indexed parameter tables, also get an
analytic gradient, so they are estimable under **NUTS** as well as IF2/PF.

Two cases are newly constrained:

- **Structural data cannot be a parameter (compile error).** Interpolation
  knots, piecewise step grids, and the periodic-spline basis are precomputed at
  construction and cannot vary per step, so a parameter driving one of those —
  or a parameter used as a **non-constant table lookup index** — is now a
  **compile error** (it was a silently-broken zero gradient). Use a constant, or
  a forcing whose coefficients are live (`sinusoidal`, `fourier`, `periodic`).
- **NUTS-only limitation (no error; the model compiles and runs).** A parameter
  that is a **periodic step value** or an **inline-table value reached by a
  non-constant index** evaluates live — estimable with IF2 or the bootstrap
  particle filter — but its gradient is not yet emitted, so a **NUTS** fit that
  depends on it is refused at fit time (not compile time) with a clear message.
  Full derivatives are tracked in gh#215.

**Migration.** No change for the common (now-working) cases. For the structural
compile error, make the coefficient constant or switch to a `sinusoidal`/
`fourier`/`periodic` forcing. For the NUTS limitation, estimate with IF2/PF, or
express the seasonality as a `sinusoidal`/`fourier` forcing (analytic gradient).

**Diagnostic.** Compile-time `error[E600]` "parameter '…' drives a … forcing
coefficient, which is structural data … cannot be an estimated parameter",
naming the parameter and forcing. The NUTS limitation surfaces at fit time:
"NUTS cannot estimate parameter(s) […]: each drives a forcing or inline-table
coefficient whose gradient is not yet emitted (gh#215) …".

## 2026-06-04 — phantom `output {}` sub-blocks removed

**What.** The `summary {}`, `flows {}`, `synthetic {}`, and experiment/compare
sub-blocks inside `output {}` never did anything and were removed; using them is
now an error.

**Migration.** Delete them. Trajectory cadence and format are configured on
`output {}` directly (see `camdl docs language`); there is no per-quantity
sub-block surface.

**Diagnostic.** `error[E106]` on the removed sub-block.

## 2026-05-26 — strict dimensions on likelihood arguments (gh#116)

**What.** Observation-likelihood arguments with a fixed dimensional contract —
`Binomial.p`, `Bernoulli.p`, `BetaBinomial.alpha`/`beta`,
`NegBinomial.dispersion` — are now strictly checked. A _count_ where a
probability/dimensionless value is required (the textbook missing-`/N` bug) is
rejected instead of silently accepted.

**Migration.** Make the argument dimensionless: `binomial(n = N, p = projected)`
where `projected` is a _count_ → `p = projected / N` (a proportion). A
projection that is already a proportion (`projected = I / N`) is fine.

**Diagnostic.** `error[E304]` "must be dimensionless (probability); a count here
is almost certainly a missing `/N`."

## 2026-04-22 — every forcing requires a unit-kind tag (GH #8)

**What.** A forcing declaration must carry a unit-kind literal after its type,
so the compiler knows whether the forcing is a count, a rate, a ratio, etc. The
un-annotated form no longer parses.

**Migration.**

```
forcing {
  pop    : interpolated { ... }      →   pop    : interpolated 'count { ... }
  birthrate : interpolated { ... }   →   birthrate : interpolated 'per_year { ... }
  school : periodic { ... }          →   school : periodic 'ratio { ... }
}
```

Same for `sinusoidal`/`piecewise`. Pick the kind from what the forcing _is_ (a
population is `'count`, a multiplier is `'ratio`); see the forcing-kinds
taxonomy in `camdl docs language`.

**Diagnostic.** `error[E001]: syntax error` at the forcing type (no migration
hint yet — see the policy in CLAUDE.md; this log is the bridge until the
diagnostic points here directly).

## 2026-03-28 — `functions {}` renamed to `forcing {}`

**What.** The block declaring time-varying covariates (population, birth rate,
seasonal terms) was renamed from `functions {}` to `forcing {}`.

**Migration.** Rename the block keyword: `functions {` → `forcing {`. The
contents are unchanged (modulo the unit-kind tag added 2026-04-22, above).

**Diagnostic.** `error[E001]: syntax error` on the `functions` keyword.

---

_This log is seeded with the breaking changes surfaced so far; older or smaller
changes may not yet be backfilled. Add an entry (on top) whenever a breaking
language change lands — see CLAUDE.md, "Breaking language changes must signpost
the migration."_
