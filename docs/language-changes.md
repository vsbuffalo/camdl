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
