---
date: 2026-06-10
status: proposal — design review (no code yet)
supersedes: 2026-06-06-observation-system.md
related:
  - 2026-06-10-multi-stream-multi-cadence-union-axis.md # the inference-side companion (proposal B)
  - 2026-06-09-burnin-conditioning-window.md # condition_from rides this binding
area: observation data entry / DSL surface / binding
issue: gh#171 (sparse/irregular obs), gh#98 (date-parse equivalence), gh#172
---

# Observation data entry: the `~` surface, typed columns, explicit indexing, aggregation

## 0. What this is, and the governing principle

This proposal fixes the **DSL surface and the binding layer** for observation
data: how a stream declares its file columns and their types, how the
measurement model is written, how data is indexed by **time and strata**, and
how data **coarser than the model** is aggregated. It unifies the file-reading
substrate that tables, forcings, and observations each duplicate today.

**Governing principle — no implicit mapping, no positional mapping; everything
explicit.** A `.camdl` file declares every column, every dimension, and every
role explicitly; the binding matches **by name**; nothing is positional and
nothing is inferred. This is a deliberate correction: the observation loader
took "column 0 is time" and fell back to a positional value column — the G1 bug
— and it is exactly the silent-wrong-answer class this software cannot afford.
The principle drives the **observation and forcing** surfaces here (§6);
**tables are a deliberate exception in v1** — their dimension columns already
bind by level name and their value shape is type-signature-validated (no
G1-class hole), so full table-binding-by-name is named as a separate follow-up,
not improvised here (§6.2).

It is the **data layer**. The **inference-time** handling of streams on
different cadences (the union time axis + per-observer flow reset) is the
sibling proposal `2026-06-10-multi-stream-multi-cadence-union-axis.md` (proposal
B); this proposal feeds it (§7) but does not implement it. Survey
**denominators** are not a special construct — they fall out of the typed-column
design (§3).

**Required reading:** `docs/camdl-language-spec.md` §6 (tables), §7 (forcings,
the `time_col`/`value_col` convention), §12 (observations, the `[p in dim]`
indexed form), the likelihood namespace (§14.2); `parser.mly` (the `TILDE` prior
productions ll. 161–185, the `index_bindings_opt` form l. 447/463); the binding
types in `rust/crates/sim/src/inference/multi_stream_obs.rs`; `dimcheck.ml` (the
Binomial/Poisson arms ~l. 843). High-risk: touches inference math and a
load-bearing DSL surface.

## 1. The problem

Observation data enters camdl through a surface that is **inconsistent with the
rest of the language, under-typed, implicitly/positionally bound, and silently
aggregating**:

1. **The likelihood is the lone holdout from `~`.** Priors are
   `beta : rate in [0.01, 2.0] ~ log_normal(...)` — `~` already means "is
   distributed as." Observations write `likelihood = neg_binomial(...)`. Two
   notations for one idea.
2. **Per-observation auxiliary data has no home.** The **k-of-n family** — test
   positivity, seroprevalence, variant frequency, environmental-surveillance
   positivity, age/space positivity — is Binomial/BetaBinomial with an
   **external denominator `n` that varies per observation and is data, not a
   parameter**. Today `Observation { time, value }` carries one value; the
   binomial's `n` is an `Expr` (constant/param), so a varying, data-driven `n`
   cannot be expressed. Same gap for Poisson person-time **offsets** and
   per-observation reporting fractions.
3. **The binding is implicitly/positionally bound and was silently wrong.** The
   value column fell back to a positional index on a header typo (G1, now
   fixed); the **time column is still positional** ("column 0 is time",
   `pfilter.rs:813`); tables map columns **positionally** (`spec §6`). Three
   observation loaders, a table reader, and a forcing reader each re-implement
   read-file-and-map-columns; date parsing is duplicated Rust-side across them
   (gh#98 is the cross-language pin, a separate matter — §6).
4. **Cross-strata aggregation is automatic and silent.** An un-indexed
   `incidence(infection)` on a stratified model compiles to `CumulativeFlowSum`
   and **sums across all strata** (`multi_stream_obs.rs:157`).
5. **`simulate` and `fit` are asymmetric.** `simulate` already emits
   multi-cadence, multi-stream, stratified data (one TSV per stream); the
   inference loaders reject heterogeneous schedules (the inference-time half is
   proposal B).

## 2. The observation block surface

### 2.1 `~` for the measurement model

The likelihood is written with `~` — the operator the language already uses for
"is distributed as." The **left side is a value column** declared in
`columns { }` (§2.2); the right side is the distribution:

```camdl
observations {
  weekly_cases from flu {
    columns       { time : time, cases : count }
    projected     = incidence(infection)
    cases         ~ neg_binomial(mean = rho * projected, r = k)
    emit_schedule = every 7 'days     # simulate-only (§2.5); fit reads the `time` column
  }
}
```

`weekly_cases from flu` names the stream (`weekly_cases`) and the **data
source** it reads from (`flu`); `from <label>` is the binding seam covered in
§2.4. The header is `name from <label> { }` — **no colon** (the block brace
follows the header directly, as `interventions`/`events` blocks do).

`cases ~ neg_binomial(...)` reads as the statistical model and parallels the
parameters block (`beta ~ log_normal(...)`). Rules that keep it unambiguous:

- **The `~` RHS is keyword-only** (`binomial(n = …, p = …)`), exactly as
  `likelihood =` is today — no positional distribution args (else
  `binomial(tested,
  …)` is ambiguous about which declared count column is
  `n`).
- **The likelihood `~` production does NOT carry the prior's `| dim` pooling
  suffix.** The prior grammar (`parser.mly:194`) allows `~ dist(...) | age`
  (hierarchical pooling over a dimension); that is meaningless on a likelihood,
  so the observation `~` is a _distinct_ production sharing the `TILDE` token,
  and `cases ~ dist(...) | age` is a hard error pointing at `[a in age]`
  indexing.
- Prior-vs-likelihood is disambiguated by the enclosing block (`parameters` vs
  `observations`); the trade vs the unmistakable-but-uglier `likelihood =`
  keyword is deliberate — one operator, two block-scoped uses.

**The projection stays a separate field** (`projected = incidence(...)`):
inlining it into the `~` would bury the incidence-vs-prevalence classification
(which drives the flow-reset semantics) inside a distribution expression. So `~`
replaces `likelihood =` and nothing else.

### 2.2 `columns { }` — the full file schema, explicit, by name

A stream declares **every column its data file carries**, with a role, in one
block. Per the governing principle there is **no implicit default and no
positional binding**: `columns { }` is always present and lists time, the index
(`: dim`) columns, and the value columns, matched to the file headers **by
name**:

```camdl
es[p in patch] from es_data {
  columns {
    time     : time         # the time axis (exactly one) — the FIT time source
    patch    : dim          # a model dimension; values validated against patch levels
    positive : count        # observed value (the ~ LHS)
    tested   : count        # auxiliary value (the denominator, referenced on the RHS)
  }
  projected     = prevalence(I_shed[patch = p]) / (baseline + rain(t))
  positive      ~ binomial(n = tested, p = detect(projected))
  emit_schedule = every 14 'days     # simulate-only (§2.5)
}
```

`columns { }` is the file's _schema_, visible to humans and agents; roles come
from the declaration (`time` / `dim` / a value type) and from the `~` statement
(LHS = scored outcome, RHS references = auxiliary). Validation cross-checks (all
require the file header, which the reader provides — §6):

- exactly one `: time` column; file headers must match the declared names
  exactly (force-matching; no positional fallback, no rename-on-bind in v1);
- **every file column is accounted for** — a header present in the file but
  absent from the `columns { }` of _any stream reading that source_ is an error
  (not silently dropped — that is how a forgotten stratum column becomes a
  silent partial-coverage miss). The accounting is **per source**: when two
  streams share `from flu`, the union of their declared columns must cover the
  file header (§2.4);
- the `~` LHS is a declared value column; every declared value column is the LHS
  or RHS-referenced (no dead columns; no undeclared LHS).

Value types (`count`, `real`, `probability`, …) are the existing parameter
types; the type lets the dimchecker verify the likelihood argument the column
feeds (§3.1).

**Binding key vs column mapping.** The **source label** (`from <label>`) is the
data **key**: `--data flu=flu.tsv` (`args/mod.rs:1245`) and
`[data.observations] flu = "…"` (`config_v2.rs:292`) bind the _source_ to a file
(§2.4). `columns { }` governs only the _within-file_ mapping. The source label,
not the stream name, is what a file binds to — so several streams can read one
file, and the **column-completeness check (§2.2) is per-source, not
per-stream**.

### 2.3 `projected =` and the incidence/prevalence axis

`projected` is the model quantity scored against the outcome column:

- `incidence(flow)` — a flow accumulated between observations
  (`CumulativeFlow`), the accumulator resetting after each observation
  (`Interval`/`TemporalKind`);
- `prevalence(comp)` / a state expression — read at the instant (`Instant`); no
  accumulation, no reset.

Orthogonal to per-observation aux (§3): serosurvey = prevalence + denominator;
positivity = incidence + denominator; census = prevalence + pure.

### 2.4 `from <label>` — the data-source seam

A stream's header is `name from <label> { … }`. The `<label>` names a **data
source** — the thing a file binds to. This separates two concerns the old
"stream name _is_ the data key" surface conflated:

- the **stream** is a model-side declaration (a projection, a likelihood, a
  cadence) — there can be many;
- the **source** is a file (`--data flu=flu.tsv`,
  `[data.observations] flu = "…"`) — one file, possibly feeding several streams.

`from` is **optional**; omitting it defaults the source label to the stream
name, so the common 1:1 case is unchanged in spirit:

```camdl
observations {
  cases { … }            # source label defaults to `cases`; --data cases=cases.tsv
}
```

The motivating N:1 case — a single wide file carrying two outcome columns scored
by two different measurement models — is now expressible without breaking the
per-source column-completeness rule (§2.2):

```camdl
observations {
  cases  from flu { columns { time : time, cases  : count } … cases  ~ … }
  deaths from flu { columns { time : time, deaths : count } … deaths ~ … }
  # source `flu` (--data flu=flu.tsv) carries time, cases, deaths;
  # the union of the two streams' columns covers the header → OK.
}
```

Without `from`, each of `cases`/`deaths` would demand its own file and each
file's _other_ outcome column would trip the completeness check. `from <label>`
is also the **forward-compatible hook** for the deferred "declare the file
schema once and refer to it" feature (§8): a named source is the natural place a
reused `columns { }` schema would later attach. The 1:1 default keeps that
complexity out of the common case.

The header form is `IDENT index_bindings_opt (FROM IDENT)? LBRACE … RBRACE` —
**no colon**. (Was `name : { }`; the colon is dropped — see migration §9.)

### 2.5 `emit_schedule` — simulate-only emission, not a fit constraint

`emit_schedule` (renamed from the old `schedule` / `every`) is the **emission**
schedule: it tells `camdl simulate --obs` _when_ to emit synthetic observations
for this stream. It plays **no role in fitting** — verified against the code:
the inference path builds observation times from the data file's `time` column
and never reads the declared schedule (the inference loaders even stub it to
`ObservationSchedule::AtTimes(vec![])`), and the per-stream incidence-bin
cadence is derived from the _observed_ times (the modal gap), not the schedule.
So for data the times are matched exactly from the `time` column;
`emit_schedule` is ignored.

Consequences:

- **Optional.** A model you only ever _fit_ omits `emit_schedule` entirely; it
  is required only to `simulate --obs` (generate synthetic data). That
  optionality is the clearest signal it is not a fit-time constraint.
- It takes a schedule literal: `every N 'unit` (regular cadence) or
  `at [t1, t2, …] 'unit` (explicit times) — the IR's `Regular | AtTimes`.
- The fit time source is unambiguously `columns { time : time }`; the emission
  schedule is a distinct, simulate-side field. The old name `schedule` — and
  jamming `every` onto the `projected` line — wrongly implied a fit-time role.

```camdl
# fit-only: no emit_schedule needed; the data's `time` column drives everything
weekly_cases from flu {
  columns   { time : time, cases : count }
  projected = incidence(infection)
  cases     ~ neg_binomial(mean = rho * projected, r = k)
}
```

## 3. Per-observation auxiliary data — declared, typed value columns

The denominator `n` in `binomial(n = tested, p)` references the declared column
`tested`. The design is sound precisely because the columns are **declared and
typed** in `columns { }`, not free-floating names.

### 3.1 Why declared, not a free expression — and the dimcheck work this requires

Every name in a likelihood expression is resolved, dimension-checked, and
differentiated **at compile time in OCaml**, before any file is read. A free,
load-time column reference breaks all three; a declared column fixes them:

- **Name resolution.** A declared, stream-scoped column resolves unambiguously;
  a collision with a parameter/compartment/forcing/level name is a **hard error
  naming both** (vs a silent re-binding).
- **Dimensional checking — including a real fix.** **Today the binomial `n` is
  NOT dimension-checked**: `dimcheck.ml` (~l. 843) does `ignore (infer … b.n)`
  and only constrains `p`; BetaBinomial `n` and the Poisson rate are likewise
  inferred-and-discarded. So "declaring `tested : count`" alone does not verify
  the denominator today. **This proposal adds the missing `constrain_known` on
  the Binomial/BetaBinomial `n` (to `count`) and the Poisson rate/offset** as
  in-scope work, so a declared `tested : count` feeding `n` is actually checked.
  (Until that lands, the column type drives Rust-side cell typing +
  autodiff-constant marking only — stated honestly, not over-claimed.)
- **Autodiff.** A declared data column is `∂/∂θ = 0`; the gradient path already
  treats binomial `n` as constant (`obs_model.rs:206`), so a declared column
  slots in with no new gradient machinery.
- **Per-row presence.** Declaring value+aux columns together lets `bind` enforce
  "outcome and its referenced covariates present together, or it is a hole" — so
  `binomial(n = NaN)` (a missing denominator poisoning resampling) is
  unconstructible.

This is the pomp `covar`/`covartable` idiom made type-safe — not a magic
`Counted` keyword (too narrow), not a free expression (unsound).

### 3.2 Aux roles — one mechanism, distinct likelihood semantics; where the guard lives

The declared-column mechanism carries every role; the semantics live in the
likelihood family:

- **Denominator that caps the outcome** — Binomial/BetaBinomial `n`. The
  **per-row data check `value ≤ n` and `n > 0` is enforced in `bind`, with a
  located row number** (a malformed survey row is a _data_ error caught at load,
  not a fit-time `-Inf`); the _projected-vs-n_ relationship stays a guard in the
  family eval. A binding cap that yields `-Inf` does not poison resampling
  (`normalize_log_weights` handles a non-finite weight), but the gradient is 0
  there while the value is `-Inf`, so the gradient-consistency test must include
  a near-`k = n` boundary point.
- **Offset that scales a rate** — a Poisson person-time column
  (`rate = lambda * projected, offset = person_time`); `k` is unbounded. A
  declared column expresses it; a `Counted` pair cannot.
- **Per-observation covariate** — an externally supplied reporting fraction or
  normalizer (`mean = report_frac * projected`), per-round assay Se/Sp. Same
  mechanism.

The existing `diagnostic_test(...)` family is **not a separate denominator
case**: `parser.mly:488-525` desugars it at parse time into a binomial whose `p`
is reparameterized by sensitivity/specificity (`p → se·p + (1−sp)·(1−p)`). Its
`n` is the same binomial `n`, so the `value ≤ n`/`n > 0` row check and the
declared-column denominator apply to it for free — no extra surface.

### 3.3 Deferred, not foreclosed

- **Multinomial / compositional outcomes** (serotype/variant splits sharing a
  denominator) are a **vector-valued scored outcome** — the one axis that _is_ a
  cell-variant extension (`ObsCell = Scalar | Vector`), distinct from aux: a
  multinomial's shared denominator is still an aux column (§3.2), and the
  `Vector` reserved variant is for the outcome vector only. The multinomial
  likelihood itself is deferred. (There is **no** `Counted` variant — a
  per-observation denominator is an aux column carried by `columns { }`, not a
  cell shape; see §6.1.)
- **Censoring** (detection limits, reporting triangles) — changes the
  likelihood's functional form; deferred, named loudly.

## 4. Indexing — strata × time, explicit, by name

### 4.1 Header-form `[p in dim]` — the language-wide idiom

A stratified observation uses the **same `name[p in dim]` indexing idiom the
rest of the language uses** (transitions `infection[p in patch]`, compartments,
`let`s):

```camdl
cases[p in patch, a in age] from flu {
  columns {
    time  : time
    patch : dim
    age   : dim
    cases : count
  }
  projected     = incidence(infection[patch = p, age = a])
  cases         ~ neg_binomial(mean = rho * projected, r = k)
  emit_schedule = every 7 'days     # simulate-only (§2.5)
}
```

The brackets bind the loop variables `p`, `a` for the **whole block** (both
`projected` and the `~` are evaluated once per `(patch, age)` cell with `p`, `a`
bound) — consistent with how `infection[p in patch] : S[p] --> E[p]` binds `p`
for the transition body. The bracket form was preferred over a parallel `by =`
keyword (which would have been a second, colliding strata mechanism) and over
indexing on the `~` line (which would leave `projected`'s use of `p`/`a`
unbound). A non-stratified stream omits the brackets; its name is a clean data
key.

**Index the projection by name.** The projection writes
`infection[patch = p, age = a]` — each model dimension named, the loop variable
on the right. This matches the governing principle (no positional mapping) and
is unambiguous when a transition is multiply stratified. The bare positional
form `infection[p, a]` is **also accepted** — it is not a silent-rebind risk,
because each subscript is dimension-membership-validated at expand time (`p`
must range over `patch`, `a` over `age`; an out-of-dimension subscript is a
located error, spec §12.1) — but the named form is preferred for readability and
is what the examples use. This is a style/consistency point, not a correctness
gate.

**Two declarations, distinct jobs, cross-checked (not redundant):**
`columns
{ patch : dim }` declares the _file column_ and validates it;
`[p in patch]` binds the _loop variable_ for expressions. The compiler
cross-checks them both ways: every `[_ in X]` needs an `X : dim` column (no data
to stratify by, else error); a `: dim` column with no bracket index is flagged
(the data is finer than the model iterates — you meant to aggregate, §5, or
forgot a bracket). `time` is never a bracket index (it is the axis, declared
`time : time`).

### 4.2 By-name level matching, and what "bins differ" does

A `: dim` column's **values are matched to the model dimension's levels by
name** (a camdl dimension _is_ a categorical factor; its levels are the bins).
By-name is a correctness requirement — a positional match would silently score
one stratum's data against another compartment. From it:

- **same bins** (file `age ∈ {0-4, 5-14, 15+}` = model levels) → fine;
- **different cutpoints** (file `{0-5, 6-17, 18+}`) → **hard error: "unknown
  level `0-5` in column `age`; model `age` levels are [0-4, 5-14, 15+]"** —
  never a silent remap (mapping `0-5` onto `0-4` would misassign data). Re-bin
  upstream;
- **data coarser than model, nested** (file `{child, adult}`, model
  `[0-4, 5-14, 15+]`) → the same unknown-level error, fixed by an explicit
  aggregation (§5) — "coarser age data" is §5's 1:many aggregation on the age
  dimension;
- **finer / non-nested** → re-bin upstream (data prep); not silently inferred.

Unknown level = located error; a model level absent from the file = that
`(time, level)` cell is **unobserved** (a hole — no term, no false zero), which
is the normal partial-coverage serosurvey shape.

**No `_col` override and no implicit defaults in v1.** A file whose headers do
not match the declared names is renamed upstream (a transparent, auditable data
step). An external-name → internal-name remapping is a deferred cross-cutting
feature (§9), not a per-column escape hatch here.

## 5. Aggregation — data coarser than the model

### 5.1 What the auto-sum actually is

Auto-summing across strata decouples the aggregation from the reporting model.
Crucially, for **uniform** reporting the auto-sum `rho * incidence(infection)`
and the explicit `rho * sum(p in patch,
incidence(infection[p]))` compute the
_identical_ value — so the gate below is a **forcing-function to make the
reporting-level decision explicit, not a fix for a wrong number in the uniform
case**. The genuinely-wrong case is _non-uniform_ reporting written as uniform:
the honest quantity is `sum(p in patch, rho[p] * incidence(infection[p]))`, and
applying a single `rho` outside the sum is silently wrong.

### 5.2 Require explicit aggregation when the data is coarser than the model

The gate is decidable at `bind` (a `CumulativeFlowSum` over >1 flow on a stream
whose data is coarser than the family). A bare un-indexed projection that would
silently sum across strata, on a stratified model, is a **hard error** naming
the choice:

```
error: stream `national_cases` is national, but the model is stratified by `patch`.
       An incidence projection over a stratified family would silently sum all
       patches and apply reporting uniformly. State the aggregation explicitly:
         • uniform reporting:   cases ~ ...( rho * sum(p in patch, incidence(infection[p])) )
         • per-patch reporting: cases ~ ...( sum(p in patch, rho[p] * incidence(infection[p])) )
```

This forces the modeler to confront _where reporting applies_ and which strata
are covered. It has a known false-positive class — a model stratified for
transmission with genuinely national data and genuinely uniform reporting is
_correct_ and is asked to write the longer `rho * sum(...)` (the same value) —
which is why this is a "make the decision explicit" gate, accepted as
loud-by-design (D3).

**Sub-national 1:many aggregation is deferred.** District model + state data
needs an aggregation map (`district × state`, itself a table) plus a partition
check (each district in exactly one state) and a precise "`by`-bound index
inside `sum`" binding — a sub-feature with its own correctness surface. v1
supports the national (all-to-one) case via explicit `sum(...)`; sub-national
rollup is named in §9.

## 6. The binding seam — one validated reader for obs + forcings

### 6.1 `BoundObs`, typed columns, and where aux lives (no `Counted`)

`bind(streams) -> Result<(BoundObs, BindReport), BindReport>` stays the single
validated constructor every path routes through. Two axes, kept separate:

- **The scored outcome** is the `ObsCell` variant — `Scalar` today, `Vector`
  reserved for a multinomial outcome (§3.3). That is the _only_ cell-variant
  axis.
- **The per-observation aux** (a binomial denominator, a Poisson person-time
  offset, a reporting fraction — zero, one, or several) is **not** a cell
  variant. It is carried by the `columns { }` binding: a stream's bound cells
  carry the scored value, and the referenced aux columns are bound per
  observation alongside, read by the likelihood **by name** (`n = tested`,
  `offset = person_time`). `bind` enforces present-together-or-hole (value and
  every referenced aux present, or the cell is a hole) and the row checks
  (`value ≤ n`, `n > 0`).

This **replaces the old `ObsCell::Counted { value, denom }`** — that variant
conflated the outcome with a single, hardcoded "denom" aux, which (per §3.2)
cannot express an offset or a second aux column. `columns { }` subsumes it: the
denominator is just one declared aux column. The strata index (§4) and
aggregation (§5) resolve in `bind`, where the model's dimension levels and the
file header are both in hand.

### 6.2 Unify the reader for obs + forcings; keep the concepts distinct

The shared, bug-prone substrate is **read a delimited file → map columns to
typed roles → validate → index**. Extract one core:

```rust
read_long(file, role_policy, time_opts) -> Result<Vec<TypedRow>, BindError>
//   separator-from-extension, comment/blank skipping, header detection,
//   one Rust-side date parser, finiteness guard.
```

**`role_policy` is `ByName` for observations and forcings** — the two surfaces
with the G1 silent-wrong class (a positional value-column fallback on a header
typo). Unifying their reader removes G1 and the loaders' duplication.

**Tables are deliberately _not_ folded into this in v1.** A table's binding is
not the G1 class, for two reasons verified against the code, not the spec:

- a table's **dimension columns are already matched by name** to the dimension's
  levels — the spec §6.2 line that says "positional mapping" (spec:890) is
  **stale doc-vs-code**; `dimcheck`/expand validate dimension membership by
  level name (spec §12.1:763). Correcting that stale sentence is doc hygiene
  done with this work, not a code change;
- a table's **value shape is validated against its declared type signature** (an
  N×M contact matrix must present N×M values); a shape mismatch is a located
  error, not a silent misread. So there is no positional-fallback hole to close.

Full **table-binding-by-name** is a real but **separate** follow-up (§8) with
its own correctness surface that this proposal must not improvise:
inline-literal tables (no file, no headers to match), a contact matrix indexed
by the **same dimension twice** (`contact[age, age]` — two columns would need
disambiguated names, not just the dimension name), and wide-form matrix files
(one column per level). Bundling those into "by-name everywhere" is how a
one-line principle turns into an unscoped refactor. v1 leaves tables on their
shape-validated binding and names the by-name table work as deferred.

**Scope correction:** this does **not** kill the gh#98 battery. That test is a
_cross-language_ OCaml↔Rust contract (OCaml parses date literals in source at
compile; Rust parses data files at load); a Rust-side reader cannot remove the
OCaml parser, so the OCaml↔Rust caltime golden stays. `read_long` unifies the
Rust-side parsing only. (And the by-name-time flip — §1 problem 3 — must land
_with_ the reader extraction, or the shared core re-introduces a positional
default.)

Do **not** unify the concepts above the reader: `tables` (compile-time,
dimension-indexed, RHS coefficient), `forcings` (compile-time, time-indexed,
**interpolated** RHS), `observations` (load-time, grid-indexed, scored LHS,
**never interpolated**) stay distinct. Forcings interpolate; observations must
not — a unified "indexed data source" with an interpolation knob would put
_imputing a missing observation_ (instead of marginalizing it) one autocomplete
from the obs loader.

### 6.3 Missing-token policy

`bind` takes a per-stream missing-token set (default `{NA}`, extensible:
`missing = {NA, -, .}`) mapping listed tokens to holes; unlisted non-numeric
tokens are hard errors; censored tokens (`<5`) stay a loud reject pending §3.3.

### 6.4 Which checks live in OCaml (compile) vs Rust (bind) — and what the IR must carry

The two-stage architecture (OCaml expand → IR → Rust simulate/fit) splits this
proposal's validation, and the split decides what the IR contract must carry:

- **OCaml, at compile (file not yet seen):** the `~` parse and
  prior-vs-likelihood disambiguation (§2.1); `columns { }` _internal_ coherence
  (exactly one `: time`, `~` LHS is a declared column, no dead columns); the
  `[p in dim]` ↔ `: dim` cross-check (§4.1); the dimcheck of likelihood
  arguments **including the new `constrain_known` on Binomial/BetaBinomial `n`
  and the Poisson rate/offset** (§3.1). For that constraint to bite, the
  declared column must be registered in the dimcheck environment as a `Known`
  count — declaring `tested : count` is what registers it; a column declared
  `real` would not constrain `n`. The aggregation gate (§5.2) is **also
  compile-side** (it is decidable from the model stratification + the stream's
  index, both known pre-file).
- **Rust, at `bind` (file in hand):** the file-header force-match (§2.2); the
  per-source column-completeness check (§2.4); **by-name level matching** (§4.2)
  — matching a `: dim` column's _values_ against the model dimension's _level
  names_; the per-row `value ≤ n`/`n > 0` data check (§3.2); missing-token →
  hole (§6.3); partial-coverage holes (§4.2).

**IR contract consequence — `observation_model` gains fields (the atomic
OCaml+Rust+golden update).** Today `observation_model` requires only
`{name, schedule, projection, likelihood}` (`schema.json:649`) — no column
schema, and `schedule` doubles as both the simulate-emission cadence and the
(ignored-in-fit) declared times. This proposal extends and clarifies it, split
across the two stages (§10):

Stage 1 (surface + binding):

- rename `schedule` → `emit_schedule` — simulate-emission only (§2.5); the fit
  time source is the `columns` `Time` column, not this field;
- add `source: String` — the `from <label>` data key (defaults to `name`);
- add `columns: [{ name, role: Time | Dim(dim) | Value(type) }]` — the file
  schema the Rust loader binds by name (the `Time` column is the fit time
  source);
- add `scored: String` — the `~` LHS column the likelihood scores.

Stage 2 (rich data):

- add `levels: [String]` per `: dim` column — the ordered level set, so the Rust
  binder can match a `: dim` column's _values_ against the model dimension's
  level names (§4.2); today the expanded IR flattens strata into per-cell names
  without the labels;
- represent the denominator/aux column references in the likelihood (binomial
  `n = <col>`), so the bind layer resolves them against the per-observation aux
  columns (§6.1) — no `Counted` cell variant.

Both stages bump `ir/VERSION` and break every golden — the atomic
OCaml(`ir/`)+Rust(`ir/src/`)+golden update per "Changing the IR schema." Flagged
here so it is not discovered mid-implementation.

## 7. Two evaluation paths must agree; and the spine connection

**Sample/score/mean/gradient.** Scoring
(`log_likelihood_from_flows_and_counts`), emission (`sample`/`mean`,
`multi_stream_obs.rs:708/730`), and the gradient evaluate the likelihood's
argument expressions through _four_ hand-mirrored functions. A state- or
forcing-dependent argument (a detection probability over a compartment and
`rain(t)`) must evaluate identically in all four — the gh#6 class. The
cross-path agreement test must be a **matrix over (likelihood family) × (aux
role) × (path)**, not a single point, and the `dt = 0.0`-at-obs-boundary literal
(present in all four contexts) must be one shared `EvalCtx` constructor so a
`dt`-referencing forcing in a likelihood cannot silently read `0` in one path.

**The spine connection (clean).** The scheduling substrate (`schedule.rs`)
builds a `Schedule` from `dt`, a `StepPolicy`, and a **sorted list of
observation boundaries** (`Schedule::with_obs`, `:187`); `build_substep_grid`
turns that into the substep grid + substep→obs map. The binding here produces
exactly that input shape (a stream's observation times), so it rides the
existing `with_obs` seam with no change to the spine — and proposal B's **union
axis is just the sorted _merge_ of all streams' times**, the same shape, fed to
the same `Schedule`. The per-stream membership + per-observer reset is a layer
on top of the substep walk, not a change to it. The recent scheduling-spine work
deliberately decoupled the observation layer from the spine, which is what makes
this a drop-in.

## 8. Scope and deferrals

- **In:** the `~` surface; `columns { }` full explicit schema (always required);
  `from <label>` data-source binding; per-obs aux as declared typed columns;
  header-form `[p in dim]` indexing with by-name level matching and
  partial-coverage holes; explicit national aggregation; the `read_long` reader
  unification **for observations + forcings**; extensible `ObsCell`;
  missing-token policy; the dimcheck-`n` fix.
- **Companion (proposal B):** multi-cadence union axis + per-observer reset.
- **Deferred, named:**
  - **Declare a file schema once, reuse it** across streams/sources (the
    `from <label>` seam is the forward-compatible hook — §2.4); v1 repeats
    `columns { }` per stream.
  - **`columns { }` auto-derive / inference** from the file header — v1 is
    verbose by design (the explicit schema is documentation for humans and
    agents).
  - **Right-truncation / nowcasting** (reporting triangles — the highest-value
    deferral for real surveillance; an observation seen "so far" is a censored
    count) and other **censoring** (detection limits).
  - **Time-varying Se/Sp supplied as a column** (per-round assay
    sensitivity/specificity beyond the per-obs covariate of §3.2).
  - **Cumulative-reporting observations** (data is a running total, not a
    per-interval increment — a different accumulation contract).
  - **Full table-binding-by-name** (inline-literal, same-dimension-twice,
    wide-form matrix files — §6.2).
  - external-name → internal-name remapping (`<=` / `=>`, cross-cutting with
    tables/forcings); multinomial / shared-denominator outcomes; sub-national
    1:many aggregation; **time-free summary-statistic observations** (final
    size, peak timing — a different observation _kind_, no time index);
    wide-form stratified files (one column per level — common Excel export;
    long-form only in v1); a stateful environmental reservoir under
    chain-binomial inference (QSS-derived-expression is the fittable interim;
    gh#191 is the real blocker).

## 9. Migration (breaking; alpha, signposted)

Each breaking change rejects the old form with a diagnostic naming the
replacement (and a `docs/language-changes.md` entry):

- `likelihood = D(...)` → `<value_col> ~ D(...)`. The `projected` quantity is
  now referenced from the `~` RHS (`mean = rho * projected`) rather than read
  out of a `likelihood =` expression — the `Projected` expr node's valid scope
  moves to the `~` RHS (spec §14.2), `projected =` stays its own field (§2.1).
- **stream header colon dropped:** `name : { … }` → `name { … }` (and the new
  optional `name from <label> { … }`). Diagnostic names the rewrite.
- **`every`/`schedule` → `emit_schedule`** (§2.5): the emission cadence is
  renamed and clarified as simulate-only (`emit_schedule = every 7 'days`); for
  fitting it is optional and ignored (the `time` column drives). Diagnostic
  names the rewrite.
- positional/implicit column binding → required `columns { }` with by-name
  headers; positional "column 0 is time" → `time : time`.
- un-indexed cross-strata auto-sum on a stratified model → the §5.2 hard error.
- **doc-only:** correct the stale spec §6.2 "positional mapping" sentence
  (spec:890) — table dimension columns already bind by level name (§6.2). No
  code or golden change; tables' binding is unchanged in v1.

## 10. Implementation in two stages

Two clean, independently-landable breaking changes. **Stage 1** delivers the new
surface + binding (the headline `~` + typed columns + by-name file reading) and
migrates the 32 `.camdl` fixtures + goldens **once** to the final scalar shape.
**Stage 2** adds the rich data layer (survey denominators + stratified
semantics). Both bump `ir/VERSION`.

### Stage 1 — the surface + binding (IR change: `emit_schedule` rename + `source`/`columns`/`scored`)

- **`~` surface + header reshape:** parser production reusing `TILDE` (without
  the `| dim` suffix; keyword-only RHS); header drops the colon and gains
  optional `from <label>` (`IDENT index_bindings_opt (FROM IDENT)? LBRACE`);
  `emit_schedule` replaces `every`/`schedule` (§2.5). Migration diagnostics for
  each.
- **`columns { }` (always required)** + the `read_long` by-name reader for obs +
  forcings (the by-name-time flip; tables untouched, §6.2); `from <label>`
  source resolution + per-source column-completeness.
- **IR schema (atomic OCaml+Rust+golden, §6.4):** rename `schedule` →
  `emit_schedule`; add `source`, `columns`, `scored`. Scalar values; keep
  today's `[p in dim]` indexing; **no** denominators / by-name level matching /
  aggregation yet.
- **Migrate** the 32 `likelihood =` fixtures + ocaml/golden models to the new
  form; regenerate `ir/golden` + `ir/expected` (reviewed).
- Tests: G1 unconstructible; positional-time gone; a header absent from the
  source's union of `columns { }` errors; two streams sharing one source bind;
  `emit_schedule` ignored in fit (data times drive); existing fits recover
  params unchanged.

### Stage 2 — the rich data layer (IR change: `levels` + denominator column-refs)

- **Per-obs aux / survey denominators** — `binomial(n = tested)` etc. carried as
  declared aux columns bound per observation (§6.1, **not** an
  `ObsCell::Counted` variant); the dimcheck-`n` fix (`constrain_known`); the
  `value ≤ n` row check; the IR carries the likelihood's column references.
- **By-name level matching** for `[p in dim]` indexed streams (the `levels` IR
  field, §4.2): per-cell scoring, partial-coverage holes, bins-differ errors.
- **National aggregation** — the §5.2 hard error; explicit `sum(...)` forms.
- **Cross-path agreement matrix** (§7) + the shared `dt`-boundary `EvalCtx`.
- Tests follow the matrix (family × aux-role × path) — no single-point coverage.

## 11. References

- Aux-role taxonomy + the k-of-n family — pomp (King, Nguyen & Ionides 2016
  JSS), spatPomp, EpiNow2/epinowcast, FluNet, GISAID, WHO polio AFP+ES,
  Rogan–Gladen (1978), Hui–Walter (1980).
- Auto-sum today: `multi_stream_obs.rs:157`. Dimcheck-`n` discard:
  `dimcheck.ml ~843`. Positional time: `pfilter.rs:813`. Positional tables:
  `spec §6`. Prior `~` grammar + `| dim`: `parser.mly:161-194`. Indexed-obs
  form: `parser.mly:463`. Forcing `_col` convention: `spec:1093`. Two-eval-path
  / gh#6: `obs_model.rs`, `multi_stream_obs.rs:708/730`. Spine seam:
  `schedule.rs:187`, `pgas.rs build_substep_grid`. gh#98 cross-language pin.
- Superseded: `2026-06-06-observation-system.md` (carried forward:
  bind-not-join, `Option` holes, the NaN guard, the positional-fallback removal
  — extended here).
