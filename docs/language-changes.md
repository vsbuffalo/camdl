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

## 2026-08-05 — a declaration's index binder must name a declared dimension

**What.** `infection[a in aeg]` where `aeg` is not in `dimensions { }` is
rejected with `E263`. This covers every construct that carries an index binder —
transitions, `let`s, init entries, observations, interventions, events,
forcings, reactive interventions, quantities.

This is a **narrowing**, and the reason it matters is the quiet case. An
undeclared axis used to DROP OUT of the expansion product, so the declaration
expanded as if it had no index at all. What you saw depended on whether the body
mentioned the binder.

If it did, you got two errors naming identifiers you never wrote:

```camdl
dimensions { age = [child, adult] }
infection[a in aeg] : S[a] --> I[a] @ beta * S[a] * I[a]
```

```
error[E100]: undeclared name 'I_a'
error[E100]: undeclared name 'S_a'
```

`I_a` is the internal mangled name — `a` was never bound, so `I[a]` mangled the
binder itself into the compartment name. Neither error named the typo.

If the body did **not** mention the binder, there was no diagnostic at all:

```camdl
infection[a in aeg] : S --> I @ beta * S      # compiles clean
```

```
$ camdlc a6.camdl | jq '.model.transitions[].name'
"infection"                        # not infection_child, infection_adult
```

One unstratified transition where the author asked for a family — a stratified
model quietly running an unstratified one.

_Migration._ Correct the axis name, or declare it. The diagnostic names the
declared dimensions:

```
error[E263]: 'aeg' is not a declared dimension
  = hint: the index binder `… in aeg` needs a dimension; declare it in
          `dimensions { aeg = [...] }`, or correct the name —
          declared dimensions: age
```

**What is unchanged.** A _declared_ dimension that happens to have no levels is
a different condition and still drops out of the product; the check keys on "not
declared," never on "no levels."

Increment A6 of `docs/dev/proposals/2026-07-31-aggregation-semantics.md`.

## 2026-08-05 — a reduction binder may not reuse a declared name

**What.** `sum(x in dim, …)` where `x` is also a declared parameter,
compartment, `let` binding, forcing, or table is rejected with `E283`, located
at the binder.

This is a **narrowing**, though the old behaviour was already an error — just
the wrong one, pointing somewhere else. `E283` owned binder shadowing but its
scope was the four _binding_ sources (a nested binder, the transition binder, a
stream index, an enclosing shaped-`let` index). A binder colliding with a
declared global fell outside it, and what you got instead was a claim about a
dimension level:

```camdl
parameters { beta : rate  gamma : rate }
@ beta * S[a] * sum(gamma in age, gamma * I[gamma]) / N[a]
```

```
error[E100]: undeclared name 'adult'
```

`adult` is declared — it is a level of `age`, right there in `dimensions {}`.
The binder substitutes each level in for `gamma`, so the body becomes
`adult * I_adult`, and the substituted identifier resolved against nothing. The
hint ("add a declaration in compartments/parameters/…") pointed at a fix that is
not one: declaring a compartment named `adult` does not help.

_Migration._ Rename the binder:

```camdl
@ beta * S[a] * sum(b in age, gamma * I[b]) / N[a]
```

```
error[E283]: reduction binder 'gamma' shadows the declared parameter 'gamma'
  8│ … * sum(gamma in age, gamma * I[gamma]) / N[a]
   │       ~~~~~~~~~~~^
  = hint: rename the binder — it may not reuse the name of a parameter,
          compartment, `let` binding, forcing, or table.
```

**What is unchanged.** A binder named after a dimension **level**
(`sum(child in age, …)`) is still legal — it does not produce this failure, and
`W103` already covers the `let`-versus-level case as a warning.

gh#495.

## 2026-08-05 — a table declared over an undeclared dimension is now an error

**What.** Every axis in a `tables { }` declaration must name a dimension
declared in `dimensions { }`. An undeclared axis is rejected with `E263`,
located on the axis list.

This is a **narrowing**, and a sibling of the `sum(...)` change below: the same
"an unknown axis has no levels" collapse, one declaration site further up. It
used to be caught only at a _use_ — indexing the table with a level in the bad
axis — so a table never indexed on that axis compiled clean:

```camdl
dimensions { age = [child, adult] }
tables { C : age × aeg = [[1.0,2.0],[3.0,4.0]] }   # `aeg` is a typo for `age`
```

Two things rode on the silence. The inline cell-count check (`E202`) skips any
table with a zero-size axis, so `C` above was never checked against its declared
shape either. And a forcing whose `time_dim` is the phantom axis lowered to an
interpolation with **zero knots** — which evaluates to 0 at every time:

```camdl
tables  { temp_data : patch × week = [[1.0,2.0,3.0],[4.0,5.0,6.0]] }
forcing { temperature[p in patch] : interpolated 'ratio {
            table = temp_data  time_dim = week  method = linear } }
```

compiled to `{"times": [], "values": []}`. The Rust loader did reject that, but
late and with no source location.

_Migration._ Correct the axis name, or declare it. The diagnostic names the
declared dimensions:

```
error[E263]: 'aeg' is not a declared dimension
  6│ tables { C : age × aeg = [[1.0,2.0],[3.0,4.0]] }
   │              ~~~~~~~~~^
  = hint: declare it in `dimensions { aeg = [...] }`, or correct the name —
          declared dimensions: age
```

**What is unchanged.** A declared dimension with no levels is not this error,
and an axis that is declared but does not match the table's shape keeps its own
diagnostics (`E202` for the cell count, `E229` for a forcing's `time_dim`). One
undeclared name now produces one `E263` even when several sites can see it; the
declaration is the one you get.

gh#490, gh#491.

## 2026-08-04 — `sum(...)` over an undeclared dimension is now an error

**What.** A reduction whose axis is not declared in `dimensions { }` is rejected
with `E263`. It used to compile silently and evaluate to `0.0`, because an
unknown axis and a `where` guard that selected no levels both resolved to an
empty domain — and an empty reduction is legitimately zero.

This is a **narrowing**: a model with a typoed axis compiled before and does not
now. That is the point. In a force of infection the folded zero removed
transmission entirely and the model still ran, producing a flat epidemic that
looks like a result:

```camdl
dimensions { age = [child, adult] }
@ beta * S[a] * sum(b in aeg, I[b]) / 100.0   # `aeg` is a typo for `age`
```

was `beta * S_child * 0.0 / 100.0`. In an observation projection it was worse —
`projected = sum(b in aeg, incidence(infection[b]))` lowered to a literal
`const 0.0`, so a fit scored every observation against an expected count of
zero.

_Migration._ Correct the axis name, or declare it. The diagnostic names the
declared dimensions:

```
error[E263]: 'aeg' is not a declared dimension
  7│ … @ beta * S[a] * sum(b in aeg, I[b]) / 100.0
   │                       ~~~~~~~^
  = hint: declare it in `dimensions { aeg = [...] }`, or correct the name —
          declared dimensions: age
```

**What is unchanged.** An empty domain that comes from a `where` guard is still
legal and still zero — an isolated patch with no in-radius neighbour contributes
nothing. That case now warns (`W202`) rather than erroring, and only when the
guard is what emptied it. A declared dimension with no levels is likewise not
this error.

gh#488.

## 2026-07-27 — bare stratified `transfer(from = S, to = V)` expands per stratum

**What.** A `transfer` endpoint may now name a bare stratified compartment. The
action expands to one atomic transfer per cell, paired cell-for-cell, in every
block that carries actions — `interventions {}`, `events {}`, and
`reactive_interventions {}`. With `age = [child, adult]`,
`vacc : transfer(fraction = cov, from = S, to = V) at [1]` emits
`FractionTransfer(S_child, V_child, cov)` and
`FractionTransfer(S_adult, V_adult, cov)`.

This is a **widening** — the bare form was previously rejected with `E264`, so
no model that compiled before is rejected now. The explicitly indexed form
(`vacc[a in age] : transfer(from = S[a], to = V[a], …)`) is unchanged, as are
single-cell transfers across strata (`from = S[child], to = S[adult]`) and into
an unstratified compartment (`from = S[child], to = V`).

**Four forms that used to report `E264` now report a specific code.**

- Endpoints with different shapes — one stratified and one not, different
  dimensions, or the same dimensions in a different order — are **E237**. The
  check compares _declared dimension vectors_, so two dimensions sharing level
  names (`age = [low, high]` vs `risk = [low, high]`) do not silently pair.
- `count =` on a bare stratified transfer is **E238**: a count is absolute, so
  fanning it out would move `count` individuals out of _every_ stratum.
  _Migration._ `transfer(count = n, from = S, to = V)` →
  `transfer(fraction = f, from = S, to = V)`, or
  `vacc[a in age] : transfer(count = n, from = S[a], to = V[a])`.
- A bare endpoint inside an **indexed family** is **E239**: it would fan out
  within each instance, transferring every cell once per instance. _Migration._
  `vacc[a in age] : transfer(from = S, to = V)` →
  `vacc[a in age] : transfer(from = S[a], to = V[a])`, or drop the binder.
- A bare **staged** compartment (`via erlang(...)`, spec §9.4) as an endpoint is
  **E237** rather than `E264`. _Migration._ `transfer(to = E)` →
  `transfer(to = E_s1)`.

---

## 2026-07-12 — indexed parameters accept multiple dimensions

**What.** A parameter declaration may now index over more than one dimension,
expanding to one scalar parameter per cell of the cartesian product:

```camdl
parameters {
  mu[village, season]         : rate               # → mu_kwaru_wet, mu_kwaru_dry, …
  m[village, season, species] : positive 'ratio     # → m_kwaru_wet_gam, …
}
let C[v in village, s in season] = mu[v,s] * m[v,s,gam]
```

This is a **widening** — no migration needed. A single-dimension declaration
(`R0[patch]`) is unchanged, and every existing model compiles identically. Cell
names mangle `<base>_<level1>_<level2>_…` in declaration-dim order; a use
(`mu[v,s]`), a scenario key (`mu_kwaru_wet`), and an init/`--param-vec` override
all address a specific cell by that name. Each index axis must name a declared
`stratify` dimension.

**Diagnostics.** A repeated axis (`mu[village, village]`) is **E331**; an
unknown or empty index dimension is **E330**; under-indexing a use (`mu[v]` on a
2-D param) remains **E299**.

## 2026-07-10 — forcing selectors quoted, `method` a bare enum, `until` → `to`

Three surface changes land together (gh#423). All are signposted: the old form
is rejected with a diagnostic that names the new form.

**What (1) — forcing file columns are quoted strings.** In a `forcing { }` block
the file selectors — `data`, `time_col`, `value_col`, `key_col` — must now be
**quoted strings**; a bare identifier is rejected. This carries the reader's
rule in the syntax: **quoted = outside the model (a file or a file column), bare
= inside the model** (a parameter, table, dimension, or a closed enum). The
model-name arguments `table` and `time_dim` stay **bare** (they name a
`tables {}` entry / a dimension), and quoting them is now rejected. The compiled
IR is unchanged — selectors are consumed at expand time and never cross the IR
seam.

_Migration._ `time_col = t` → `time_col = "t"`; `value_col = C` →
`value_col = "C"`; `key_col = village` → `key_col = "village"`; `data = f.tsv` →
`data = "f.tsv"`. Leave `table = mymatrix` and `time_dim = week` bare.

_Diagnostic._ A bare file selector → **E410**
(`value_col must be a quoted
column name — write value_col = "C"`). A quoted
`table`/`time_dim` → **E412**
(`table names a model construct — remove the quotes`).

**What (2) — `method` is a bare, validated enum.** The interpolation method is
now written **bare**: `method = linear` (also `constant`, `spline`), matching
every other closed enum in the language (`integrator = rk45`, `format = tsv`).
The quoted form is rejected, and — new — the value is **validated**: only
`linear | constant | spline` are accepted. Previously an unquoted-vs-quoted
`method` compiled either way and an unknown value (`method = "cubic_spline"`)
was accepted by the compiler but failed to deserialize at the simulation
boundary. The spec's `cubic_spline` / `pchip` never existed in the runtime and
are removed.

_Migration._ `method = "linear"` → `method = linear`. If you used
`"cubic_spline"` or `"pchip"`, switch to `spline`.

_Diagnostic._ A quoted `method` → **E411**
(`method is now a bare enum — write
method = linear`); an unknown value →
**E411**
(`unknown interpolation method
'banana' — expected linear, constant, or spline`).

**What (3) — the recurring window end is `to`, not `until`.** In a
`transfer(...) { … }` / `add(...) { … }` recurring schedule body, the window end
was spelled `until`, while a `set` block and `simulate { from … to … }` spelled
the same slot `to`. `to` is now the only spelling.

_Migration._ `{ every = 30 'days  from = 0 'days  until = 90 'days }` →
`{ every = 30 'days  from = 0 'days  to = 90 'days }`.

_Diagnostic._ `until = …` → **E113** (`` `until` is now `to` ``).

---

## 2026-07-05 — ASCII `*` accepted as an alias for the `×` dimension separator

**What.** The dimension-product separator — in table shapes
(`C_age : age × age`) and typed `let` shapes (`: patch × patch`) — now accepts
the ASCII `*` as an exact alias for the Unicode `×`. `age * age` compiles
identically to `age × age`; the separator is purely syntactic (it names the
axes), so the choice never affects the IR. This mirrors rate expressions, where
`×` and `*` are already interchangeable as multiplication.

**Migration.** None — additive and backward-compatible. Every existing model
compiles unchanged. `×` stays canonical and is recommended in committed models
for readability; `*` is a hand-authoring escape hatch for keyboards without the
glyph.

**Diagnostic.** None — the grammar is strictly more permissive. Previously
`age * age` in dimension position produced **E001** (syntax error); it is now
accepted.

---

## 2026-06-29 — reserved no-overlay scenario sentinel renamed `as_fitted` → `fitted`

**What.** The reserved scenario name for the no-overlay row in
`camdl fit
predict` output — the fitted model with no scenario overlay applied,
the value carried in the leading `scenario` column — is renamed from `as_fitted`
to `fitted`. The name is reserved: a `scenarios { }` preset may not use it,
because it would shadow the no-overlay value and make output rows ambiguous.

**Migration.** If you parse `camdl fit predict` output, the no-overlay
`scenario` column value is now `fitted` (was `as_fitted`). The reservation moves
with the name: a `scenarios { }` preset may no longer be named `fitted` (rename
it); a preset named `as_fitted` is no longer reserved.

**Diagnostic.** A `scenarios { }` preset named `fitted` → **E291** ("scenario
name 'fitted' is reserved"), with a hint to rename the scenario.

---

## 2026-06-26 — partial dimension omission in a rate read is now an error

**What.** Indexing a compartment with _some but not all_ of its dimensions in a
rate expression is now a hard error. A compartment stratified over 2+ dimensions
— e.g. `E` declared over `[age, latent_stage]` — referenced as `E[a]` (dropping
`latent_stage`) has no defined cell: the compiler cannot tell whether you meant
a specific stage or their sum. Previously this silently fell through to
name-mangling and produced a confusing `E100` against a synthetic compartment
the user never wrote (`undeclared name 'E_adult'`), with no source location.

Unchanged: the **bare name** `E` (no brackets) still sums over _all_ dimensions,
and a **full index** `E[a, e1]` still resolves to one cell. Only the
partial-index middle case is newly rejected.

**Migration.** Index every dimension (`E[a, e1]`), or — to fix one dimension and
sum over another — marginalize **explicitly**:

```
# old (now E287):  gamma * E[a]
# new:             gamma * sum(s in latent_stage, E[a, s])
```

**Diagnostic.** A partial index in a rate read → **E287** ("compartment 'E' has
dimensions [age, latent_stage] but only 1 of 2 were indexed; a partial index has
no defined cell"), located at the index node, with a hint giving both the
full-index and explicit-`sum(...)` forms.

---

## 2026-06-18 — `scope` key removed from `reactive_interventions {}` (gh#204)

**What.** The `scope = exogenous | particle` reactive key is removed. A reactive
trigger always reads **reported surveillance** — the realized observation draw,
shared across particles. The `particle` (latent-state) scope was never wired: it
parsed but the runtime ignored it, silently behaving as `exogenous`, so it is
withdrawn until latent-scope triggers are actually implemented (at which point
the key returns with the inference-safety seam that is its reason to exist). (IR
schema 0.17 → 0.18.)

**Migration.** Delete the `scope = ...` line — `exogenous` is the only behavior
and is now implicit.

**Diagnostic.** `scope = ...` in a reactive policy → **E106** ("the `scope`
reactive key was removed … remove it, exogenous is implicit").

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
  }
}
```

The `when` predicate is a boolean over the trigger inputs `observed(stream)` and
`sum_observed(stream, window = D)`, combined with `and` / `or` / `not`. Reactive
policies lower to the IR's new `fire = Reactive(..)` source — internally the
intervention `schedule` field became `fire: Scheduled | Reactive`, an orthogonal
axis from `kind` (a reactive policy is `kind = Scenario, fire = Reactive`).
Forward chain-binomial runs the agenda; Gillespie/ODE and inference reject an
active reactive policy with a `REACTIVE_INTERVENTIONS` capability error. (IR
schema 0.16 → 0.17.)

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
**E272**; stream-arg / window arity **E270** / **E271**. Running a model with an
_active_ reactive policy → a `REACTIVE_INTERVENTIONS` capability error (parsed
but not yet executable).

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
