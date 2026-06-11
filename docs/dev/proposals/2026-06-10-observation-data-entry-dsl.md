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

# Observation data entry: the `~` surface, typed columns, strata indexing, aggregation

## 0. What this is, and what it is not

This proposal fixes the **DSL surface and the binding layer** for observation
data: how a stream declares its data columns and their types, how the
measurement model is written, how data is indexed by **time and strata**, and
how data **coarser than the model** is aggregated. It also unifies the
file-reading substrate that tables, forcings, and observations each duplicate
today.

It is the **data layer**. The **inference-time** handling of streams on
different cadences (the union time axis + per-observer flow reset) is the
sibling proposal `2026-06-10-multi-stream-multi-cadence-union-axis.md` (proposal
B); this proposal is compatible with it and names where they meet, but does not
implement it. Survey **denominators** are not a special construct here — they
fall out of the typed-column design (§3).

**Required reading:** `docs/camdl-language-spec.md` §6 (tables), §7 (forcings),
§12 (observations), §2666 (the likelihood namespace); the current binding types
in `rust/crates/sim/src/inference/multi_stream_obs.rs`
(`BoundObs`/`bind`/`StreamProjection`/`ObsCell`, the scoring seam at :593, the
sample/mean emission paths at :708/:730); the prior `~` grammar in
`ocaml/lib/compiler/parser.mly` (the `TILDE` productions, ll. 161–185). This
touches inference math and a load-bearing DSL surface — treat as high-risk.

## 1. The problem

Observation data enters camdl through a surface that is **inconsistent with the
rest of the language, under-typed at the binding boundary, and silently
aggregating** in ways that bake in modeling assumptions. Concretely:

1. **The likelihood is the lone holdout from `~`.** Priors are written
   `beta : rate in [0.01, 2.0] ~ log_normal(...)` — `~` already means "is
   distributed as." Observations write `likelihood = neg_binomial(...)`. Two
   notations for one idea.
2. **Per-observation auxiliary data has no home.** Real surveillance is
   dominated by the **k-of-n family** — test positivity, seroprevalence, variant
   frequency, environmental-surveillance site-positivity, age/space positivity —
   all Binomial/BetaBinomial with an **external denominator `n` that varies per
   observation and is data, not a parameter.** Today
   `Observation { time, value }` carries one value; the binomial's `n` is an
   `Expr` (a constant/param), so a _varying, data-driven_ `n` cannot be
   expressed. The same gap blocks Poisson **person-time offsets** and externally
   supplied per-observation reporting fractions.
3. **The binding boundary is under-typed and was silently wrong.** Tables map
   columns _positionally_; observations map _by name_; the loaders drifted (G1:
   a single-stream loader fell back to a positional column, binding the wrong
   data on a header typo — now fixed). NaN/Inf passed through to the log-PMF
   (now guarded). Three loaders (`pfilter.rs`, `profile.rs`, `fit/runner.rs`)
   plus a table reader plus a forcing reader each re-implement
   read-file-and-map-columns; date parsing is duplicated across OCaml-compile
   and Rust-load and pinned equal only by a battery test (gh#98).
4. **Cross-strata aggregation is automatic and silent.** An un-indexed
   `incidence(infection)` on a stratified model compiles to `CumulativeFlowSum`
   and **sums across all strata** (`multi_stream_obs.rs:157`). A national datum
   then scores against the summed _true_ incidence with a single reporting
   fraction applied outside the sum — silently assuming **uniform reporting and
   full coverage** across strata, which is usually false.
5. **`simulate` and `fit` are asymmetric.** `simulate` already emits
   multi-cadence, multi-stream, stratified data (one TSV per stream); the
   inference loaders reject heterogeneous schedules. The format exists; the
   binding does not. (The inference-time half is proposal B.)

The unifying theme — flagged in `ARCHITECTURE.md` as "the consolidation that
pays down the whole observation-data tier" — is that the **binding** is the
bug-prone shared substrate, and the **surface** has drifted from the language's
own conventions.

## 2. The observation block surface

### 2.1 `~` for the measurement model

The likelihood is written with `~`, the operator the language already uses for
"is distributed as" (priors). Two reserved names anchor it: **`projected`** (the
model's predicted quantity) and **`observed`** (the data):

```camdl
observations {
  weekly_cases : {
    projected = incidence(infection)
    every     = 7 'days
    observed ~ neg_binomial(mean = rho * projected, r = k)
  }
  detection : {
    projected = prevalence(I)
    every     = 14 'days
    observed ~ bernoulli(p = p_detect)
  }
}
```

`observed ~ neg_binomial(...)` reads as the statistical model, parallels the
parameters block (`beta ~ log_normal(...)`), and collapses two notations into
one. **The projection stays a separate field** (`projected = incidence(...)`):
inlining it into the `~` would bury the incidence-vs-prevalence classification
(which drives the flow-reset semantics) inside a distribution expression. So `~`
replaces `likelihood =` and nothing else; `projected`/`every`/`by`/`columns`
remain their own lines.

For a single-value stream, `observed` is the stream's one data column. When a
stream has multiple data columns (§3), the `~` LHS is the **outcome** column by
name (`positive ~ binomial(...)`), and the others are referenced on the RHS.

### 2.2 `columns { }` — the file's typed schema

A stream declares the **value columns** its data file carries, with types, in
one block:

```camdl
es : {
  every   = 14 'days
  columns { positive : count, tested : count }
  positive ~ binomial(n = tested, p = detect(projected))
}
```

The block is the file's _schema_; roles are assigned by the `~` statement (LHS =
outcome, RHS references = auxiliary). This co-locates types (you read the file's
shape in one place), mirrors how a table declares its columns, and keeps the `~`
line a clean statistical statement. A `aux`/`observe` role keyword was
considered and rejected: once `~` assigns roles, a role keyword double-marks.

**Validation cross-check** (a feature of separating schema from role): every
column in `columns { }` must be the `~` LHS or referenced on the RHS (no dead
columns); the `~` LHS must be a declared column (catches typos both ways).

`count`, `real`, `probability` etc. are the existing parameter types; a column's
type is what lets the OCaml dimchecker verify the likelihood argument it feeds
(`n = tested` requires `tested : count`).

**Binding key vs column mapping (what changes, what does not).** The stream
_name_ is the data **key** — `--data es=es_data.tsv` (`args/mod.rs:1245`,
`NAME=PATH`) and `[data.observations] es = "es_data.tsv"` (`config_v2.rs:292`,
an `IndexMap<name, path>`) bind the observation block `es` to a file, matched to
`model.observations[].name` (`runner.rs:303`). That is unchanged and already
wired. What `columns { }` changes is the _within-file_ mapping: today the loader
reads the single column **named after the stream**
(`load_observations(path, column = stream_name)`, `runner.rs:301`), so a stream
`weekly_cases` must have a `weekly_cases` column — the single-column assumption
that blocks a positivity file (`positive` + `tested`, with no `es` column).
`columns { }` declares the file's columns explicitly, **decoupling them from the
stream name**: `--data es=es_data.tsv` then reads a `positive`/`tested` file.
The key (which file) stays the stream name; the columns (which fields inside it)
become declared.

> **Open decision D1b — single-column default.** For a stream with one value
> column and no `columns { }`, keep today's "the column is named after the
> stream" default (back-compat, terse), or require a one-line
> `columns { weekly_cases : count }`? My recommendation: **default the single
> undeclared column to the stream name** (the common case stays a one-liner) and
> require `columns { }` the moment there is more than one column — so the
> multi-column case is always explicit and the single-column case is never more
> verbose than today.

> **Open decision D1 — block name.** `columns { }` vs `data { }`. `data { }` is
> intuitive but collides with fit.toml's `[data.observations]` (file paths). My
> recommendation: **`columns { }`** (literal, no collision).

### 2.3 `projected =` and the incidence/prevalence axis

`projected` is the model quantity scored against `observed`:

- `incidence(flow)` — a flow accumulated between observations
  (`CumulativeFlow`); the accumulator resets after each observation (the
  `Interval`/`TemporalKind` axis). Reset semantics across cadences are proposal
  B.
- `prevalence(comp)` / a state expression — read at the observation instant
  (`Instant`); no accumulation, no reset.

This axis is **orthogonal** to per-observation aux (§3): a serosurvey is
prevalence + a denominator; positivity is incidence + a denominator; a hospital
census is prevalence + pure. Both axes compose.

## 3. Per-observation auxiliary data — declared, typed, stream-scoped

The denominator `n` in `binomial(n = tested, p)` references the per-observation
column `tested`. The design choice that makes this **sound** is that such
columns are **declared and typed** (in `columns { }`), not free-floating names.

### 3.1 Why declared, not a free expression

It is tempting to say "`n` is just an expression that references a data column."
That is **unsound**, because every name in a rate/likelihood expression today is
resolved, dimension-checked, and differentiated **at compile time in OCaml**,
long before any data file is read. A free, load-time column reference breaks all
three:

- **Name resolution.** A data column `tested` and a parameter `N` would compete
  for the same namespace with no defined precedence — a silent re-binding. The
  malaria positivity example uses `N_tested` as a _parameter_ today; migrating
  it to a column while the param is still declared is exactly the collision.
  Stream-scoped _declared_ columns resolve unambiguously and a name collision
  with a global is a **hard error** naming both.
- **Dimensional checking.** The dimchecker never sees the file; an undeclared
  column has no dimension, so `binomial(n = tested)` cannot be checked. A
  declared `tested : count` can.
- **Autodiff.** The gradient graph is built at compile time; a data column is
  `∂/∂θ = 0`, but only if the compiler _knows_ it is data and not a parameter. A
  declared column announces this; a free name does not.
- **Per-row presence.** If `n = tested` and `tested` is missing on a row whose
  outcome is present, a free reference yields `binomial(n = NaN)` — the exact
  NaN-into-resampling failure the value-cell guard prevents, reintroduced at the
  auxiliary column. Declaring the columns lets `bind` enforce "outcome and its
  referenced covariates are present together, or it is a hole" — unconstructible
  otherwise.

So per-observation aux is a **typed, stream-scoped declaration**
(`columns { }`), and the likelihood references it by name. This is the
established pomp `covar`/`covartable` idiom made type-safe. It is neither a
magic `Counted` keyword (too narrow — see §3.2) nor a free expression (unsound).

### 3.2 Aux roles — one mechanism, distinct likelihood semantics

Per-observation aux is not one shape. The declared-column mechanism carries all
of them; the _semantics_ live in the likelihood family:

- **Denominator that caps the outcome** — Binomial/BetaBinomial `n`. The cap
  `observed ≤ n` and `n > 0` are enforced **in the binomial family's eval** (not
  a cell type), so the guard holds whether `n` is a column, a param, or a
  constant. (This is where the stashed `Counted{value, denom}` prototype's
  discipline lives — generalized.)
- **Offset that scales a rate** — a Poisson person-time/exposure column
  (`rate = lambda * projected, offset = person_time`); `k` is _unbounded_ (not a
  cap). A single `Counted` pair cannot represent this; a declared column does.
- **Per-observation covariate** — an externally supplied reporting fraction or
  normalizer the mean references (`mean = report_frac * projected`), or
  per-round assay sensitivity/specificity. Same mechanism.

`n`-from-data and `n`-as-expression both work, because both are expressions; one
happens to reference a declared column. A fixed survey size is `n = N0` (param);
a real survey is `n = tested` (column).

### 3.3 Deferred, but not foreclosed

- **Multinomial / compositional outcomes** (serotype/variant splits,
  age-distribution-of-cases). `ObsCell` must be designed **extensible**
  (`Scalar | Counted | Vector`) so the _type_ does not wall this off, even
  though the multinomial likelihood is deferred.
- **Censoring** (detection limits, reporting triangles/nowcasting). This changes
  the likelihood's _functional form_ (an integral over a region), not just an
  argument — a different kind of feature. Deferred and named loudly.

## 4. Indexing — strata × time

### 4.1 `every` (time) + `by` (strata), by-name, per-cell

A stratified observation declares its time cadence with `every` and its strata
with `by` (the keyword `stratify(by = ...)` already uses):

```camdl
sero : {
  every = 365 'days
  by    = age                       # long-form file carries an `age` column
  columns { seropos : count, sampled : count }
  seropos ~ binomial(n = sampled, p = R[age] / N[age])
}
```

Long-form file:

```
time        age     seropos  sampled
2024-01-01  child   30       100
2024-01-01  adult   12       200
```

`by = age` binds `age` as the stream's stratum index (like a transition's
`[p in patch]`); each row binds `age` and is scored independently
(`30 ~ binomial(100, R[child]/N[child])`). Multi-strata is `by = age × region`.
A non-stratified stream omits `by` (index is time only).

**Three by-name mappings**, all by name, not position:

1. time index — a column named `time` (override §4.2);
2. strata index — the `by` column's values matched to the model's dimension
   **levels**;
3. value columns — `columns { }` matched to file headers.

**The time column is by-name too — fixing a residual positional binding.** Today
the value column is by-name (the G1 fix), but the _time_ column is still
**positional**: the loader takes column 0 (`pfilter.rs:813`, "first column is
always time"), ignoring its name. That is the same fragility class as the G1
value bug — reorder the columns so time is not first and the wrong column is
silently read as time. This proposal makes the time index **by-name** like the
other two: a column named `time` by convention, the `time = <col>` override
(§4.2) when it is named otherwise, and **no positional fallback** (a file with
no `time` column and no override is a located error, not a silent column-0
bind). So all three indices — time, strata, value — resolve by name with
overrides, and the last positional binding is gone.

By-name strata matching is a **correctness requirement**: a positional match
would silently score one stratum's data against another compartment — the worst
silent-wrong-answer. From it, two loud behaviors fall out:

- **unknown level** (file `age = infant`, model `[child, adult]`) → hard error
  naming the valid levels;
- **missing level** (model has `elderly`, the survey did not cover it) → that
  `(time, stratum)` cell is **unobserved** (a hole — no term, no false zero).
  This is the normal partial-coverage serosurvey shape, handled by construction.

### 4.2 Convention with explicit override

The default is convention: a file column named after the dimension carries that
stratum; `time`/value columns are named to match. An explicit override covers
files that do not follow it:

```camdl
sero : {
  time  = report_date              # the time index is the `report_date` column
  by    = age from age_band        # the `age` dimension is carried by `age_band`
  columns { seropos = n_pos : count, sampled = n_tested : count }
  seropos ~ binomial(n = sampled, p = R[age] / N[age])
}
```

Every mapping has a by-name default and a `role = file_column` override:
`by = age` ≡ `by = age from age`; the time index is a column named `time`,
overridden by `time = <col>` (this replaces today's positional "first column is
always time" — §4.1); value columns default to their `columns { }` names,
overridden by `stream_name = file_column`.

> **Open decision D2 — convention vs always-explicit.** My recommendation:
> **convention-with-override** (terse common case; by-name matching keeps the
> correctness-critical strata→levels mapping safe regardless; the override
> handles messy ministry files). The conservative alternative is to require the
> `from`/`= col` clauses always (no convention) — defensible for data you do not
> control, but boilerplate that gets copy-pasted wrong.

## 5. Aggregation — data coarser than the model

### 5.1 The risk

Auto-summing across strata (§1.4) is risky because it **decouples the
aggregation from the reporting model**, which cannot be decoupled. If reporting
varies by stratum (`rho[patch]`), the honest national observation is

```camdl
observed ~ neg_binomial(mean = sum(p in patch, rho[p] * incidence(infection[p])), r = k)
```

— sum the _reported_ incidence — whereas the easy auto-summing form

```camdl
observed ~ neg_binomial(mean = rho * incidence(infection), r = k)   # silently sums TRUE incidence
```

applies one reporting fraction to summed _true_ incidence: a different, wrong
quantity whenever reporting is non-uniform. Auto-sum silently assumes **uniform
reporting** and **full coverage**, and **silently re-sums** if a stratum is
added later.

### 5.2 Require explicit aggregation

"Data coarser than the model strata" is a **1:many aggregation**. National is
the extreme (all strata → one); sub-national (district model, state data) needs
an **aggregation map** (`district × state`) — which is a _table_ (the
reader/mapping machinery again). The rule:

- data at model resolution (`by = patch`, file has a `patch` column) → per-cell,
  no aggregation;
- data coarser than the model → the aggregation must be **explicit**.

A bare un-indexed projection that would silently sum across strata, on a
stratified model, is a **hard error** naming the choice:

```
error: stream `national_cases` is national, but the model is stratified by `patch`.
       An incidence projection over a stratified family would silently sum all
       patches and apply reporting uniformly. State the aggregation explicitly:
         • uniform reporting:   observed ~ ...( rho * sum(p in patch, incidence(infection[p])) )
         • per-patch reporting: observed ~ ...( sum(p in patch, rho[p] * incidence(infection[p])) )
       For coarser-than-national data, supply an aggregation map (a `district × region` table).
```

This forces the modeler to confront _where reporting applies_ and which strata
are actually covered — the load-bearing modeling decision the auto-sum hides.

> **Open decision D3 — require-explicit vs auto-sum-with-flag.** My
> recommendation: **require explicit** (hard error). The failure is a quietly
> miscalibrated likelihood (per-stratum reporting collapsed to uniform), not a
> typo — the "informs public-health decisions" stakes argue for loud. This is a
> **breaking change** to the current auto-sum (needs a migration diagnostic).
> The alternative (auto-sum + a warning naming the assumptions) preserves
> convenience but warnings get skimmed on exactly a silent-wrong-answer class.

## 6. The binding seam

### 6.1 `BoundObs`, typed columns, extensible cells

`bind(streams) -> Result<(BoundObs, BindReport), BindReport>` stays the single
validated constructor (every path routes through it). Changes:

- `ObsCell` becomes **extensible**:
  `Scalar(f64) | Counted { value, denom } |
  …` (Vector reserved for
  multinomial). The denominator/offset/covariate columns are typed fields the
  compiler knows (so `∂/∂θ = 0` and dimcheck hold — §3.1).
- a stream carries its declared **columns** (name → type) and the resolved
  outcome/aux roles from the `~` statement;
- the strata index (§4) and aggregation (§5) are resolved in `bind`, where the
  model's dimension levels and the file's columns are both in hand.

**Two resolution stages, kept distinct.** The **stream → file** key
(`--data
es=…`, `[data.observations] es = …`) is resolved at the fit-config
layer (`runner.rs`), matched to `model.observations[].name`; this is unchanged
(§2.2). The **file → columns** mapping (which column is time, which are the
value/aux/ strata-index columns) is resolved inside `bind`/`read_long`, **by
name**, with the `time = <col>` / `by = … from <col>` / `stream = <col>`
overrides (§4). The two stages are independent: the key says _which file_,
`read_long` + the `columns { }`/index declarations say _which fields in it_ — so
the same by-name-with-override discipline governs every column, and no column is
bound by position.

### 6.2 Unify the reader, not the concepts

The genuinely-shared, bug-prone substrate is **read a delimited file → map
columns to typed roles → validate → index**. Tables (positional), observations
(by name), and forcings (`time_col`/`value_col`) each re-implement it; G1 was
the observation loader drifting toward the table policy, and the gh#98
date-parse battery test exists only because OCaml-compile and Rust-load parse
dates separately. Extract one core:

```rust
read_long(file, role_policy, time_opts) -> Result<Vec<TypedRow>, BindError>
//   separator-from-extension, comment/blank skipping, header detection,
//   ONE date-column parser (kills the gh#98 battery), finiteness guard.
//   role_policy ∈ { Positional (tables), ByName (observations, forcings) }
```

Do **not** unify the concepts above this line. `tables` (compile-time,
dimension-indexed, RHS coefficient), `forcings` (compile-time, time-indexed,
**interpolated** RHS), and `observations` (load-time, grid-indexed, scored LHS,
**never interpolated**) stay three distinct role-typed declarations. The
load-bearing reason: **forcings interpolate; observations must not** — a unified
"indexed data source" with an interpolation knob puts _imputing a missing
observation_ (instead of marginalizing it) one autocomplete from the obs loader,
which is a correctness regression, not a convenience. Unifying the _reader_
removes G1 and the gh#98 battery; unifying the _concepts_ would smuggle
interpolation into the likelihood.

### 6.3 Missing-token policy

Real exports use `NA`, `-`, `.`, blank, and `<5` interchangeably. Today only
`NA` maps to a hole and everything else errors, so a routine file fails to load
with an invisible fix. `bind` takes a per-stream/per-file **missing-token set**
(default `{NA}`, extensible: `missing = {NA, -, .}`) mapping listed tokens to
holes; unlisted non-numeric tokens remain hard errors. Censored tokens (`<5`)
stay a loud reject pending the censoring feature (§3.3).

## 7. Two evaluation paths must agree

Scoring (`log_likelihood_from_flows_and_counts`, the gh#139 seam) and emission
(`sample`/`mean`, `multi_stream_obs.rs:708/730`, used for synthetic data and
posterior-predictive) evaluate the likelihood's argument expressions through
_different_ functions. A state- or forcing-dependent likelihood argument (a
detection probability depending on a compartment and `rain(t)`) must evaluate
**identically** in both — the gh#6 class (a likelihood arg scored against a
zero-scratch buffer, off by ~100×, bit twice). Per-observation covariates (§3)
and forcings-in-likelihoods widen exactly this gap, and only the scoring path is
gated. **A cross-path agreement test** (sample vs score on a state+forcing+aux
likelihood) is required, not optional, and `dt` semantics at the obs boundary
(`dt = 0.0` there, `multi_stream_obs.rs:256`) must be specified for any
`dt`-referencing forcing used in a likelihood.

## 8. Scope and deferrals

- **In:** the `~` surface; `columns { }` typed schema; per-obs aux as declared
  columns (denominator/offset/covariate); strata×time indexing with by-name
  mapping and partial-coverage holes; explicit cross-strata aggregation; the
  `read_long` reader unification; extensible `ObsCell`; missing-token policy.
- **Companion (proposal B):** multi-cadence union axis + per-observer flow
  reset. This proposal's per-stream schedules feed B's union axis; B's reset
  keys on the declared schedule, not value presence.
- **Deferred, named:** multinomial outcomes (cell type stays extensible);
  censoring (LOD/reporting triangles); a stateful environmental reservoir under
  chain-binomial inference (the QSS-derived-expression path is the fittable
  interim — the gh#191 capability gap is the real blocker).

## 9. Migration (breaking changes — alpha, but signposted)

Per `docs/language-changes.md` policy, each breaking change rejects the old form
with a diagnostic naming the replacement:

- `likelihood = D(...)` → `observed ~ D(...)` (or `<col> ~ D(...)`).
- un-indexed cross-strata auto-sum on a stratified model → the §5.2 hard error.
- (no change for non-stratified single-value streams beyond `likelihood =` →
  `~`.)

## 10. Implementation phases + tests

1. **`~` surface + `projected`/`observed`** — parser production reusing `TILDE`;
   no IR change (sugar over `Likelihood`); migration diagnostic for
   `likelihood
   =`. Test: golden models reparse; `likelihood =` rejected with
   the hint.
2. **`read_long` reader unification** — one core, `role_policy` param; route the
   three obs loaders + table + forcing readers through it; one date parser.
   Test: G1 unconstructible (positional fallback gone); gh#98 battery retired in
   favor of the single parser; missing-token set honored.
3. **`columns { }` + declared per-obs aux + extensible `ObsCell`** — typed
   columns; `n`/offset/covariate from columns; `k≤n`/`n>0` in the binomial
   family. Tests: dimcheck rejects an undeclared column; name-collision hard
   error; `binomial(n = tested)` recovers a positivity fit; person-time offset.
4. **Strata×time indexing** — `by`, by-name level matching, per-cell scoring,
   partial-coverage holes, the override forms. Tests: unknown level errors;
   missing level → unobserved cell; per-cell scoring matches a hand computation;
   `from`/`= col` override.
5. **Aggregation** — the coarser-than-model hard error; `sum(p in patch, ...)`
   explicit forms; the aggregation-map (table) path. Tests: bare auto-sum on a
   stratified model errors with the §5.2 message; explicit per-stratum-reporting
   national fit recovers params; a `district × region` rollup.
6. **Cross-path agreement** — sample vs score on a state+forcing+aux likelihood
   (§7); `dt`-at-boundary specified.

## 11. References

- The k-of-n family and the aux-role taxonomy (denominator/offset/covariate;
  censoring as a form change) — surveillance practice: pomp measles/`bsflu`
  (King, Nguyen & Ionides 2016, JSS); `spatPomp`; EpiNow2/epinowcast; FluView /
  FluNet; GISAID variant frequencies; WHO polio AFP+ES; Rogan–Gladen (1978) and
  Hui–Walter (1980) for assay correction.
- Auto-sum today: `multi_stream_obs.rs:157` (`CumulativeFlowSum`).
- The four identical-times rejections (proposal B): `fit/runner.rs:355`,
  `pfilter.rs:208`, `profile.rs:516`, `multi_stream_obs.rs:439`.
- The two-eval-path / gh#6 hazard: `obs_model.rs` scoring vs
  `multi_stream_obs.rs:708/730` emission; `dt = 0.0` at :256.
- Prior `~` grammar: `parser.mly` ll. 161–185 (`TILDE`).
- Reader/date duplication: gh#98 (the date-equivalence battery test).
- Superseded: `2026-06-06-observation-system.md` (carried forward: the
  `bind`-not-join seam, `Option`-hole missing-data semantics, the NaN guard, the
  positional-fallback removal — all retained and extended here).
