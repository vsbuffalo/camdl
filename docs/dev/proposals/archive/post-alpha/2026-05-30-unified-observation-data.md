# Unified observation data: joining, heterogeneous streams, missing data, and the burn-in boundary

Status: **SUPERSEDED (archived 2026-06-09).** This umbrella draft is replaced by
`2026-06-06-observation-system.md` (the data layer), with the
burn-in/conditioning half folded into `2026-06-09-time-interval-model.md`. Start
from `2026-06-09-time-and-observation-overview.md`. Retained for history only —
do not implement from this.

Date: 2026-05-30

**Audience:** implementers and reviewers working on the camdl observation
surface — the DSL `observations {}` block, the IR observation model, the Rust
data loader (`fit/runner.rs`, `pfilter.rs`, `caltime_load.rs`), and the
inference likelihood seam (`multi_stream_obs.rs`, `obs_model.rs`). Assumes
familiarity with the language spec (`docs/camdl-language-spec.md`, especially §6
Tables and §12 Observations) and the IR contract (`ir/schema.json`).

Originating issue: gh#134 (calendar-time ergonomics) surfaced a deeper problem
during a Kano measles fit. This proposal addresses the full observation surface,
of which gh#134 is one corner.

---

## 0. TL;DR

camdl's observation surface is currently **wide-only and single-cadence**: all
observation streams must share one identical time axis
(`multi_stream_obs.rs:308` rejects anything else), there is no missing-data path
(a `NaN` cell poisons the particle weight), and the only fitting-target shape
that works cleanly is interval- accumulation count data (incidence). This blocks
the two motivating use cases:

- **measles**: the first-window/burn-in bug (§1.1). **Scope honesty:** this
  proposal _diagnoses_ measles and ships the **guardrail** (error + opt-out) +
  missing-data plumbing; the actual covariate-informed burn-in _fix_ is the
  `t_cond` decoupling, which is **split out and pinned**
  ([`2026-05-30-conditioning-boundary-tcond.md`](2026-05-30-conditioning-boundary-tcond.md)).
  So measles is made _safe_ here, not _solved_ here.
- **malaria**: cross-sectional prevalence surveys (`k` of `n` tested) +
  ragged/sparse routine-surveillance data — the parts that fall out of the
  table-over-(time×dims) core. Age-stratified seroprevalence and EIR are **named
  but not delivered** by this proposal (§7.1) — see the scope notes there.

**Missing-data sentinel (decided):** for now, a `NaN`/empty value cell _is_ the
missing sentinel (row-absent in long form; empty or `NaN` in wide). Simple and
good enough; the typed missing-vs-zero discipline (§5.2) rides on it. (A
stricter "reject literal `NaN`, require empty cell" rule was considered and
deferred — not worth the friction now.)

The unifying design idea: **an observation is a table indexed by time plus model
dimensions.** Reuse the `read()` table/dimension column- alignment convention
(spec §6) for observation data. This single convention makes heterogeneous
cadence, missing data, cross-sectional surveys, and the synthetic-data
round-trip all fall out of one mechanism. On top of it we add three capabilities
— outer-join with per-point missing-skip, data-column likelihood arguments
(`data.<col>`), and CAS materialization of the joined table — plus one separable
inference change, the conditioning boundary `t_cond` that decouples burn-in from
the start of filtering.

This is a large lift. It is sequenced (§13) so the high-risk inference change
ships last, behind the low-risk format/loader work.

---

## 1. The problems, concretely

### 1.1 The first-conditioning-window bug (gh#134, reproduced)

The particle filter conditions on observation _k_ over the interval
`[t_{k-1}, t_k]`, propagating particles then weighting/resampling against datum
_k_. The **first** window is `[t_start, first_obs_time]`, where `t_start` comes
from the model (`simulate.from` → `Ir.t_start`, `expander.ml:3082`) and
`first_obs_time` from the data's first row (`particle_filter.rs:140,199`).

When `t_start` is far behind the first observation, this window spans the whole
gap. Reproduced on the user's Model A (single-pop Kano SEIRV) at fixed baseline
parameters, 2000 particles, seed 1, **same data**, only `simulate.from` changed:

| `simulate.from`                              | first window          | loglik       |
| -------------------------------------------- | --------------------- | ------------ |
| `date("2011-12-26")` (origin)                | `[0, 980]` = 980 days | **−3416.31** |
| `date("2014-08-25")` (1 wk before first obs) | `[973, 980]` = 7 days | **−3202.18** |

A 214-unit loglik swing at identical parameters. Two mechanisms:

1. **Window inflation** (incidence projections only): the
   `incidence(progression)` projection accumulates 980 days of new cases and
   scores it against one weekly datum → NB mean ~140× the observation.
   Prevalence projections (instantaneous reads) do _not_ inflate.
2. **Unconditioned free-run** (all projections): 980 days of seasonally-forced
   dynamics with no resampling → the ensemble decoheres and lands in an
   arbitrary seasonal phase.

**Crucial clarification — the bug is a _gap_, not zeros.** The user's own repo
contains both shapes: `kano_..._t7.tsv` (weekly rows from day 7, early values
`0`) gives a healthy 7-day first window; conditioning on early zeros is
informative and correct. `kano_..._w.tsv` (first row at day 980) gives the
broken 980-day window. The difference is the _presence of rows at cadence_ vs an
_empty leading span_. A "bunch of zeros at the beginning" is the healthy case;
an absence of rows is the bug.

**The user already hit this and worked around it.** Model A lines 69–75 set
`from = date("2014-08-25")` with the comment "NOT a 980-day accumulation from a
free, unconditioned 2012–2014 run", and `initial_susceptible_fraction` is a
_free estimated parameter_ (`fit_A.toml:16`, bounds `[0.04, 0.12]`, normal
prior) rather than covariate-derived. They _wanted_ a covariate-informed burn-in
over 2011–2014 (births accumulate susceptibles, MCV/SIA deplete them) but the
bug forced them to abandon it and estimate a free initial susceptibility
instead. That is a real loss of rigor driven by a tooling gap.

### 1.2 No heterogeneous cadence

All bound streams must share an identical time vector:

> `multi_stream_obs.rs:300`: "observation stream {} has obs_times that differ
> from stream 0; heterogeneous schedules are not supported yet"

(mirrored at `runner.rs:301`, `pfilter.rs:164`). This blocks a weekly case
series fit jointly with an annual prevalence survey, and the output side has the
symmetric limit (`main.rs:628`: "A single wide TSV cannot hold multi-cadence
streams").

### 1.3 No missing-data path

Data cells go through `parse::<f64>()` (`pfilter.rs:581`) straight into the
log-pmf at `multi_stream_obs.rs:393` with no `is_nan` guard. Rust's float parser
accepts `"NaN"`, so a NaN literal _loads_ but then poisons the particle weight →
degeneracy or silent garbage. There is no way to say "this point is unobserved."

### 1.4 Count-only fitting targets

The `Likelihood` enum (`ir/observation.rs:54`) supports Poisson, NB, Normal,
Binomial, BetaBinomial, Bernoulli — but the likelihood _arguments_ are model
expressions only. A cross-sectional prevalence survey needs the **denominator
`n` from the data** (number tested), which has no channel today. So Binomial is
declarable but unusable for real survey data.

---

## 2. Design principle: an observation is a table over (time × dims)

The spec already defines, for **tables** (§6.2):

> All file-based tables use **long format** (one row per observation, index
> columns then value column). … the compiler uses **positional mapping** from
> the type signature.

and for **data-derived dimension levels** (§6.3):

> `read(file, column = "col")` reads the named column, collects unique values in
> first-occurrence order.

An observation is structurally the same object — a value indexed by coordinates
— with one extra coordinate, `time`. So the canonical observation file is a long
TSV whose index columns are `time` plus any dimensions the block is stratified
over, and whose value column is named by the block:

```
# cases[r in region], long format
time   region   cases
980    r0       12
980    r1        3
987    r0       18
987    r1        5
```

This is exactly the shape the Kano data already ships in (`time, lga, cases`)
and exactly what DHIS2 exports. The `region` column holds level names validated
against the dimension — the **same machinery as §6.2/§6.3**, reused. The
epidemiologist learns one rule for tables, dimension derivation, and
observations.

**Named-column matching is already precedented in `read()`** (verified against
`expander.ml`, not just the spec). `read()` is _mixed_ today: table index/value
columns map **positionally** (§6.2/§6.4 — `load_table_data` reads
`List.nth cols i` by position, header used only for the W201 reorder warning),
but **dimension derivation** (`read(file,
column = "patch")`, §6.3) and
**forcing/time-function columns** look columns up **by name**
(`List.find_index (fun h -> h = col_name)`). So observations matching value
columns by name (header == block name) and index columns by dimension name
follows the _existing_ named-lookup path, not a new exception. The only thing
observations add is that the value-column header is the **block name** — which
is exactly what `simulate` already writes (§4, the round-trip). Tables stay
positional; no tables change.

---

## 3. Data layout: long and wide, reader accepts both

We do **not** force long output. Wide is easier to eyeball and `head`; long is
necessary for ragged/stratified data and is what tidy plotting wants. The reader
accepts both; the writer picks the representable one.

**Reader (auto-detect):**

- A column whose header matches a **dimension name** ⇒ long layout; that column
  carries level names (validated like §6.3).
- Extra columns whose headers match **block names** ⇒ wide layout; one value
  column per stream.
- `time` (or a date column resolved via `origin`) is always the time index.
- Ambiguity (a header matches both a dimension and a block) → hard error naming
  the collision.

**Writer:**

- **Wide when representable** (shared cadence across streams) — current default,
  preserved, round-trips. Header `time, <block>, <block>, …`.
- **Long when not** (heterogeneous cadence, or stratified families that would
  explode into hundreds of columns) — header `time, <dim>, …, <block>, …`.
- `--obs-dir` (one file per stream) stays for those who want it.

Both layouts are byte-for-byte re-readable by the fit loader.

---

## 4. The synthetic round-trip: the model is the schema

`camdl simulate` already writes observation columns named by the block
(`main.rs:1146`, `obs_stream_names.push(obs_model.name.clone())`), and spec
§12.4 confirms the same `observations {}` declaration is _sampled_ on output and
_scored_ on input. We make fit input **symmetric** to simulate output: bind
value columns by header == block name, index columns by header == dimension
name. Then the model alone defines the schema at both ends:

```bash
# generate synthetic observations from the model
camdl simulate model.camdl --params truth.toml --obs-only synth.tsv

# fit them back — zero renaming, schema derived from the same model
camdl fit run fit.toml --data synth.tsv
```

```
# synth.tsv (written by simulate; provenance header skipped on read)
# camdl synthetic | model=8f3c…  seed=42  generated=2026-05-30
time   weekly_cases
7      0
14     9
21     14
```

**Fragility guard (synthetic looks like real):** keep the data schema identical
— that _is_ the round-trip — but mark synthetic files out-of-band so a
human/agent can always tell, while a fit never mis-parses:

- a `#`-comment provenance header
  (`# camdl synthetic | model=<hash> |
  seed=<n> | generated=<date>`), skipped
  on read;
- a sidecar `synth.meta.json` (typed manifest);
- the CAS already stamps sim provenance.

**Round-trip wrinkle — survey denominators (§7).** Incidence and
prevalence-count observations round-trip cleanly. A Binomial _survey_ block
needs `n` (number tested) to _sample_ an observation, but in forward simulation
there is no data file, so `data.tested` is undefined. A survey block must
therefore supply a denominator source for the sampling direction — either a
survey-design table (`n : time × region = read(...)`) or an `n =` model
expression fallback. This is the one place the round-trip needs an extra input;
it is called out so it does not surprise.

**Stratified round-trip — long, not wide.** `simulate` today writes a stratified
family `cases[r in region]` as _expanded wide columns_
(`time, cases_r0, cases_r1, …`; `main.rs:1079` pushes the expanded
`obs_model.name`), whereas the canonical long form is `time, region,
cases`.
These are different files for the same model, so the "zero renaming" round-trip
guarantee holds **only via long format for stratified blocks.** Two required
pieces:

1. `simulate` must be able to emit **long** for stratified families (it
   currently cannot — only expanded-wide or per-file). Long is the
   round-trip-safe output for stratified models.
2. The reader must define the **demangling**
   `cases_r0 → (block=cases,
   region=r0)` if it is to accept expanded-wide as
   input at all (re-deriving the expansion). Recommendation: support
   expanded-wide read as a convenience but document long as the canonical
   stratified round-trip; do not claim renaming-free round-trip for
   expanded-wide.

---

## 5. Join semantics and missing data

### 5.1 Join mode: `union` (default) vs `strict`

(The original draft named these `inner`/`outer` after SQL joins. That was
misleading — SQL _inner_ keeps only matching keys and _drops_ the rest, the
opposite of "include all data." Renamed to behavioral terms.)

```
Join = union    # DEFAULT: union time axis; an absent (time, stream) cell is unobserved
     | strict   # opt-in: require every stream observed at every time; error on mismatch
```

**`union` is the default — include all the data the user provided.** This is the
safest default in the sense that matters most: it never silently _drops_ data,
and it makes the common multi-rate case (weekly cases + an annual survey) just
work. An absent `(time, stream)` cell is treated as unobserved (§5.2 skip).

`strict` is the opt-in for users who _want_ the safety check that every stream
is observed at every time (e.g. a synthetic round-trip where ragged data would
signal a bug). Its mismatch error is _actionable_ (names the offending stream,
shows the first divergent time on each side, points at `--join union`).

**The risk `union`-by-default introduces, and its mitigation.** Because `union`
never errors on a mismatched time, a _typo_ in a time column (day `1000` for
`100`) silently becomes an extra row that is unobserved everywhere else, rather
than an error. Mitigation (ties to the §14 Q1 guardrail, decided error+opt-out):
warn — or under the guardrail, error — when a stream's time axis is wildly out
of family with the others (e.g. its span vastly exceeds every other stream's, or
it contributes near-zero overlapping points), which is the typo/mis-join
signature. So `union` is the convenient default _with_ a guardrail against the
one failure mode it opens.

### 5.2 Missing = skip, under MAR

The likelihood seam is
`MultiStreamObsModel::log_likelihood_from_flows_and_counts`, which sums over
streams at each time. **As of gh#139 this is a single seam:** PF/IF2/PMMH reach
it via the trait `log_likelihood` (now a one-line delegate) and PGAS calls it
directly, so the skip lands once and is genuinely inherited by all four methods.
_(Pre-gh#139 there were two byte-identical summation loops — the trait method
and the flat method — and a change to one silently missed the other; that is the
GH#6 / incident-2026-04-22 ~100× divergence class. The unification was done
first, precisely so this proposal's skip/`data.<col>`/`observed≤n` work lands in
one place. Do **not** revert to per-method loops.)_

- **Stream absent at `t_k`** (no row in long form, or `NaN` cell in wide):
  contributes 0 to that step's joint log-likelihood (drop the term from the
  sum).
- **All streams absent at `t_k`**: contribute nothing to the log-likelihood and
  **do not resample** — resampling on uniform weights only injects Monte-Carlo
  variance. (Returning lik=1, as pomp does, technically works but is exactly
  this needless resample.)
- **Missing vs structural zero:** a missing row / `NaN` = skip; an explicit `0`
  = observed zero, scored. Never coerce blank→0 or skip a 0. This is the DHIS2
  zero-vs-blank trap, documented in the surveillance literature (Senegal DHIS2
  data-quality; Kenya DHIS2 completeness — see §7 refs).

**Crucial — "skip the weight" is NOT "skip the step" (resolves the apparent
contradiction with §5.2.1).** "Do not resample at an all-absent `t_k`" is a
statement about _weighting/resampling_, not about _propagation or accumulator
bookkeeping_. The filter still **stops at `t_k`** as a propagation checkpoint
and still applies the §5.2.1 accumulator-reset rule there; it just contributes
no likelihood term and does not resample. You cannot "fold `t_k` away entirely"
— that would skip the reset and re-create the window-inflation bug (§5.2.1). The
precise rule: **stop at every grid point (reset accumulators per §5.2.1); weight
and resample only at grid points where ≥1 stream is observed.** (An earlier
draft said "fold `t_k` into the next interval," which wrongly implied skipping
the checkpoint — corrected here.)

### 5.2.1 Accumulator reset is tied to the GRID, not to observation presence

**This is a correctness trap that MAR does not cover.** MAR governs _which_ time
points are missing; it says nothing about the _window semantics_ of the
surviving points. For an `incidence(...)` projection the flow accumulator
integrates over a window and resets after each observation
(`particle_filter.rs:377`). Consider a fixed-cadence weekly series with `t_k`
missing but `t_{k-1}` and `t_{k+1}` present:

- If we skip the likelihood at `t_k` **and** skip the reset, the accumulator at
  `t_{k+1}` spans `[t_{k-1}, t_{k+1}]` = two weeks.
- But a fixed-cadence weekly datum at `t_{k+1}` means "cases in the one week
  `[t_k, t_{k+1}]`."
- Scoring a two-week accumulation against a one-week datum is **the exact gh#134
  window-inflation bug, now triggered by interior missingness.**

So **skipping the likelihood term must NOT skip the accumulator reset.** The
reset is tied to the **schedule grid** (every cadence), the score to
**observation presence**. Two window semantics, declared per block:

- **`fixed_cadence` (default):** reset the accumulator at every grid point even
  when unobserved; the datum is incidence over the single cadence ending at its
  timestamp. Correct for standard weekly/monthly surveillance.
- **`cumulative_since_last_report`:** do _not_ reset at skipped points; the
  datum is cases since the previous non-missing report. Correct for some
  passive-surveillance regimes. Opt-in, because it is the rarer and more
  dangerous default.

This decoupling (score ≠ reset) is mandatory for any incidence block with
missing data and is a P1.5 deliverable (§13), not free in P1.

### 5.2.2 Per-stream reset — the multi-cadence interaction (the bigger lift)

§5.2.1 assumes a _single_ cadence per block. The headline malaria fit is
**multi-cadence**: weekly NB incidence + an annual Binomial survey on one union
axis (§5.1). That breaks the current accumulator machinery in a way §5.2.1 alone
does not cover, and it is the largest under-scoped piece of P1.5.

Today the reset is **global**: `particle_filter.rs:377` does
`for state in &mut swarm.states { state.reset_flows(); }` — _all_ flow
accumulators, unconditionally, once per scored step. The in-code canary comment
right above it (`particle_filter.rs:364-376`) already names this exact future
feature: _"If a future feature ever stores 'flow since the most recent
per-stream observation' at different cadences per stream, this reset needs to
become per-flow and indexed by which stream last observed."_ This proposal
**is** that feature.

The failure if left global: with a union axis, the annual survey's timestamp
becomes an extra grid point. If it lands _between_ two weekly points (off the
weekly cadence), the global reset zeroes the **weekly** incidence accumulator
mid-week, and the next weekly datum is scored against a partial-week
accumulation — the gh#134 window-inflation bug, re-triggered from a second
stream's cadence. Worked:

```
weekly grid:   …  196        203  …      cases ~ NB(incidence over [196,203])
survey at 200:        ↑ scored here; GLOBAL reset_flows() fires
                      → weekly accumulator zeroed at 200, so the day-203
                        datum is scored over [200,203] (3 days), not 7.
```

**Required:** `reset_flows` becomes **per-flow-accumulator, indexed by which
stream's cadence owns that flow** — reset the weekly flow only at weekly
boundaries, the survey's projection only at its own boundaries. This is a
mechanism change to `ParticleState.flow_accumulators` + `particle_filter.rs`
(and the PGAS/IF2 mirrors), **not** a per-step boolean. It is inherent to
multi-cadence incidence fitting — any tool doing this must solve it — and it is
the load-bearing correctness item in P1.5. (Prevalence/snapshot projections
don't accumulate, so they're unaffected; the problem is specifically incidence ×
multi-cadence.)

**Statistical validity — the skip is correct only under Missing At Random
(MAR).** Surveillance is frequently MNAR. Two MNAR sub-cases, and the proposal
handles only one:

- **Reporting-rate variation** (completeness known and time-varying): remediable
  with a reporting covariate in the _likelihood_
  (`mean = projected * reporting(t)`, a `forcing`/covariate — §7.3, not a
  `data.col` in a rate). This is MAR _conditional on the covariate_.
- **Outcome-dependent missingness** (a facility goes silent _because_ it is
  overwhelmed; a stockout records true cases as `0`): the missingness depends on
  the unobserved value itself. **No covariate fixes this**, and the skip biases
  the fit. This case is **out of scope** — name it plainly in user docs; do not
  imply the completeness covariate covers it.
- document the MAR assumption loudly at _runtime_ (fit summary + CAS manifest),
  not only in the spec — the affected user (stockout-prone setting) is the
  target user.

pomp's reference convention (FAQ §3.3 "How do I deal with missing data?"): the
user inserts NA rows and the measurement-model author guards `ISNA(y)` and
returns lik=1; pomp does _not_ auto-skip. Our improvement: auto-skip + typed
missing-vs-zero, so the author cannot forget the guard. (Verbatim FAQ text not
extracted yet — confirm before citing in the spec.)

---

## 6. The conditioning boundary `t_cond` (split out — PINNED)

**Decision (2026-05-30): split out and pinned.** The `t_cond` / burn-in design
moved to its own proposal,
[`2026-05-30-conditioning-boundary-tcond.md`](2026-05-30-conditioning-boundary-tcond.md),
to be revisited _after_ this observation-data surface ships. It is inference
math with an unresolved warm-up-coherence crux, and it is orthogonal to
joining/missing-data/surveys — so it should not gate them.

What stays in _this_ proposal: the **guardrail** (error + opt-out when the first
conditioning window is ≫ the modal cadence, or there is an all-missing leading
stretch). That guard is the cheap immediate safety net, independent of the full
`t_cond` machinery, and can ship with the missing-data work (P1.5, §13). The bug
it guards (gh#134) is diagnosed in §1.1; the _fix_ (decoupling burn-in from
filter start) lives in the pinned sub-proposal. Default behavior here is
unchanged: the filter still starts at `t_start = simulate.from`; the guardrail
just refuses to _silently_ run a pathological first window.

**Comparison to He et al. (book vignette `he2010_london.camdl`):** that model
uses `simulate { from = 0 'days }`, is **unanchored** (no `origin`), and its
cases + `pop(t)`/`birthrate(t)` covariates all start at `t = 0`. So
`t_start == first_obs == 0`: no gap, no bug. He has no separate burn-in window —
the entire 15-year fit window is the data window and the filter conditions
throughout; transients in the first year or two are simply conditioned away by
data that exists there. He can do that because he _has_ data spanning the whole
period. Kano cannot (no case data 2011–2014), which is exactly why `t_cond` is
needed: you cannot condition on data you do not have, but you _can_
deterministically burn in over a covariate-only span.

**Default makes the common case unchanged:** `t_cond = first_obs` (default)
reproduces today's behavior for He-style data-from-t=0 models. Kano declares a
`t_cond`/burn-in window explicitly.

This is the high-risk piece — it changes filter-start logic in
`particle_filter.rs` / `pgas.rs` / `if2.rs` and adds an IR field. It is
sequenced last (§13) and may warrant its own sub-proposal if it grows.

---

## 7. Observation kinds and the `data.<col>` likelihood argument

### 7.1 The data-type taxonomy we must support

Grounded in the malaria/measles calibration literature, split by _temporal
support_ (what the type system must model):

- **Interval-accumulation (incidence)** — case counts per week/month; the
  measles case and the dominant surveillance type. Poisson/NB with reporting ρ.
  _Already supported._
  [He-style time-series inference for malaria](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7367373/)
- **Point-in-time prevalence (cross-sectional)** — `k` positive of `n` tested at
  one time, Binomial. **The standard malaria calibration target** (*Pf*PR from
  parasitological surveys), often age-stratified. _Needs `data.<col>` for `n`._
  [community prevalence + facility data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0042861)
- **Prevalence time series** — repeated cross-sections (monthly/annual *Pf*PR),
  usually sparse/irregular. _Needs heterogeneous cadence._
- **Age-stratified seroprevalence (serocatalytic)** — Binomial
  seropositive-by-age; in endemic equilibrium `P(a) = 1 − e^{−λa}`. **NAMED, NOT
  DELIVERED by this proposal.** The catch: age = time-since-birth, so
  seroprevalence is _cohort-cumulative exposure over an age axis_ — a different
  temporal support than the `Projection` enum models (all four variants project
  state/flows at _one model time_). This is **not a new `Projection` variant**;
  it requires the _model_ to carry an age-stratified compartment whose stratum
  is an explicit `age` dimension, with the cumulative-exposure structure in the
  dynamics — then the observation is an ordinary `CurrentPop`/`DerivedExpr`
  snapshot over that age dimension and the survey `n` rides `data.col` like any
  other Binomial survey. So what's needed is a **modeling pattern (a worked
  age-structured model)**, not new observation surface. Scoped as a
  documentation/example deliverable, _not_ an engine change; resolve the worked
  model before claiming sero support.
  [Hodgson, cross-sectional serology](https://www.davidhodgson.me/post/sm5_cross/);
  [serocatalytic models](https://pubmed.ncbi.nlm.nih.gov/40696544/)
- **Entomological (EIR / vector)** — sparse spatio-temporal, heavily
  overdispersed, _continuous log-scale rate_. **OUT OF SCOPE — not expressible
  after this proposal.** The `Likelihood` enum is count/proportion only
  (`observation.rs:54`); `Normal` is the _discretized-count_ form
  (`obs_model.rs:74`), not a continuous log-Normal, and EIR is a continuous
  rate, not a flow-count or prevalence. EIR needs a new continuous/log-scale
  likelihood family — deferred to malaria-vNext, listed here only so the
  taxonomy is honest about the gap.
  [sparse spatio-temporal entomology](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4071307/)
- **Bulk / cumulative counts** ("total infections observed") — almost always an
  interval-accumulation with a wide window. There is nearly always an implicit
  time/window. _Supported_ (it's incidence).
- **Molecular FOI / multiplicity-of-infection** — the one near-time- free case
  (MOI distributions → FOI via an equilibrium relation). A likelihood on a
  derived equilibrium quantity, not the trajectory. **Deferred** — different
  evaluation path, blocks neither measles nor malaria v1.
  [MOI→FOI](https://elifesciences.org/reviewed-preprints/100076)

So the malaria unlock this proposal _actually delivers_ = **cross-sectional
Binomial prevalence + prevalence time series (heterogeneous/sparse cadence) +
missing-data skip**, all riding the union-axis core. Seroprevalence is a
_modeling-pattern_ follow-up; EIR needs a new likelihood family (deferred).
"Kind" is mostly **already expressed** by the `projected =` expression (the
`Projection` enum: `CumulativeFlow` = incidence, `CurrentPop`/`CurrentPopSum` =
prevalence). The only kind that needs new surface is the survey, because of the
denominator.

**Spatial aggregation (named gap, deferred).** The Kano work found — and
Snyder/Bengtsson/Bickel/Anderson (2008), cited §16, prove — that a
high-dimensional spatial particle filter degenerates, so fitting often _must_
aggregate spatial streams (model 44 LGAs, but score their sum). The obs design
here is **per-stratum**: `cases[r in region]` yields one likelihood term per
region (`multi_stream_obs.rs:397` sums over streams). There is **no operator to
model strata but score their aggregate**. This is a real, citation-backed need
that this proposal does **not** address — deferred to the spatial-coupling work;
named here so nobody builds a national spatial fit expecting it to be tractable.

### 7.2 `data.<col>`: the compiler/runtime split

A survey block needs two value columns bound to one block:

```
time   region   positive   tested
1095   r0            41        320
1095   r1            12        210
```

```camdl
observations {
  prevalence_survey[r in region] : {
    projected  = prevalence(I[region = r]) / pop[r]
    schedule   = from_data                  # cross-sectional: times from the data
    likelihood = binomial(n = data.tested, p = projected)
  }
}
```

**Not a new schedule kind.** `from_data` is surface syntax for the **existing**
`ObservationSchedule::FromData` (`ir/observation.rs:77`), which already means
"observation times come from the data file." The original draft wrote
`at = data`, which read as a new construct — it is not; reuse `FromData`. The
only open item (§8) is confirming `FromData` already handles _irregular_
(non-constant-cadence) times, which the cross-sectional case requires.

**Verified compiler/runtime split:**

- _Runtime (Rust): small._ `eval_likelihood_resolved` (`obs_model.rs:54`)
  already receives the scalar `observed` per observation, and `Expr` already has
  a `Projected` variant resolved from runtime context — proof the "evaluate from
  per-observation runtime input" pattern exists. Add `Expr::DataCol(name)`,
  widen each stream's `observations: Vec<f64>` to carry named auxiliary columns
  (loader already splits all header columns and discards extras), resolve
  `name → col_idx` at `MultiStreamObsModel::new` (like `Param`). Off the hot
  path.
- _Compiler (OCaml): the catch._ The OCaml frontend **never reads the
  observation data file** — tables/dimensions/forcings are read at compile time
  (`read_csv_rows`), but observation data is purely a Rust fit-time concern. So
  the compiler cannot know `data.tested` exists or its dimension. And gh#116
  (closed) made likelihood args **dimensionally checked** (e.g. `Binomial.p`
  must be dimensionless, E304; `dimcheck.ml:760+`). A `data.<col>` reference
  therefore:
  1. carries **unknown dimension** at compile time → accepted permissively (like
     an opaque external), dimensional contract on sibling args (e.g. `p`
     dimensionless) still enforced;
  2. has **existence + finiteness validated at fit time** in Rust, with an
     actionable error listing available columns. This is a deliberate,
     documented compile-vs-fit boundary.

### 7.2.1 Survey validation: `observed ≤ n` (messy-malaria guard)

The current Binomial eval (`obs_model.rs:102–107`) computes
`binom_logpmf(k, n, p)` with `k = observed`, `n = data.tested`. If a survey row
has `positive > tested` — a **common** data-entry error in real malaria survey
files — the binomial coefficient is 0 and the log is −∞, poisoning the particle
weight with **no diagnostic**. Since "messy malaria data" is an explicit target
of this proposal, this is a guaranteed silent failure.

**Required fit-time validation:** for every survey row, assert `observed ≤ n`
(and both finite, non-negative), else a hard error naming the offending file,
row, and values. This runs at data-load, before any filter time is spent.

### 7.2.2 `data` is a reserved, runtime-only namespace (LOCKED)

The denominator is referenced via a reserved `data.<col>` namespace. `data.col`
means _"the value of column `col` in the observed row for this observation's
time and stratum."_ It is unlike every other referenceable thing, and those
differences are exactly why it earns a reserved keyword — each is a rule the
compiler enforces:

1. **Runtime-only.** Params, tables, forcings are read at compile time and
   inlined into the IR. `data` is never seen by the OCaml compiler; it is bound
   at fit time. So it is **dimensionally opaque** (the compiler cannot check
   `data.tested` is a count) and its existence is checked only at data-load.
2. **Per-observation.** A forcing is a function of `t`; a table is static; a
   param is global. `data.col` varies by _both_ time _and_ stratum — the
   finest-grained input there is.
3. **Direction-asymmetric.** Always present when fitting. In simulate it exists
   only if a design supplies it; otherwise a hard error at dispatch (§7.2.3).
   Naming it `data` makes the asymmetry _visible_ in the source.
4. **Scope-restricted.** Legal **only** inside `observations {}` likelihood
   arguments. A `data.col` in a `transitions {}` rate is a compile error (new
   E-code): the _dynamics_ cannot depend on how many people were tested — only
   the _observation_ may condition on observation metadata. This is a real
   correctness guard, not just hygiene.

The conceptual split that makes the whole design close:

- **`data.X` is always an INPUT** the likelihood _conditions on_ — read in both
  directions when available.
- **the OUTCOME** (the random variable the likelihood is _about_) is _produced_
  by simulate and _scored_ by fit. It is never a `data.col` — in simulate it
  does not exist yet. The outcome is the block's value column (§7.2.4).

### 7.2.3 The ADTs (LOCKED)

The surface mirrors the IR sum types so **illegal states are unrepresentable** —
no loose optional fields validated away at runtime.

```ocaml
(* An observation block *)
type obs_block = {
  name       : ident;             (* LHS of ':' minus index — IS the outcome column *)
  index      : index_binder list; (* [r in region]; [] if unstratified *)
  projected  : expr;              (* observed model quantity; lowers to Projection *)
  schedule   : schedule;          (* WHEN — exactly one, by construction *)
  likelihood : likelihood;        (* the observation distribution *)
  column     : ident option;      (* optional outcome-column rename; default = name *)
}

and schedule =                    (* one sum type; `every`+`at` together is unrepresentable *)
  | Every    of duration          (* regular cadence over the window *)
  | At       of instant list      (* explicit times *)
  | From_data                     (* times are an observed input (data || design) *)

and likelihood =
  | Poisson       of { rate : expr }
  | Neg_binomial  of { mean : expr; r : expr }
  | Normal        of { mean : expr; sd : expr }
  | Binomial      of { n : expr; p : expr }
  | Beta_binomial of { n : expr; alpha : expr; beta : expr }
  | Bernoulli     of { p : expr }

(* expr gains ONE variant, legal ONLY inside likelihood args (rule 4 above): *)
and expr = ... | Data_col of ident | ...
```

What is deliberately **not** in the type: the observed value `k`/count. It is
not part of the model spec — it is bound at fit/sim time (the block's column).
The likelihood holds only its _parameters_. That is the clean line: the ADT
describes the distribution; the data supplies the point.

**The payoff — `From_data` and `Data_col` are the same idea twice.** One says
_the times_ are exogenous; the other says _a likelihood parameter_ is. The
"needs a design to simulate" check is therefore **one predicate over the
types**, covering both:

```ocaml
let needs_design (b : obs_block) : bool =
  b.schedule = From_data
  || List.exists is_data_col (args_of b.likelihood)
(* simulate errors iff  needs_design b && no design bound for b *)
```

The asymmetry we circled for several rounds is not scattered special-casing; it
is _one_ property — "is this input exogenous?" — that the type system sees in
both the schedule and the args.

**Surface for `schedule` (Option B, LOCKED).** One field, a tagged variant —
structurally exclusive, uniform shape, and `from_data` reads naturally in value
position (it never collides with a column name because it only appears in
schedule position):

```camdl
schedule = every(7 'days)
schedule = at([90 'days, 180 'days])
schedule = from_data
```

(Rejected: loose `every =` / `at =` / bare `from_data` fields — they let you
write `every=` and `at=` together, forcing a runtime validation that the sum
type makes unnecessary, and the three forms don't look alike. The IR's
`ObservationSchedule` enum _already_ is this 3-variant type
(`ir/observation.rs:74`) — this is a **surface-only** change to match it.)

### 7.2.4 The outcome column: name the block for what's observed; `column =` is the escape hatch

By the round-trip convention (§4) the block's value column **is the block
name**: block `weekly_cases` → outcome column `weekly_cases`. The observed value
`k` is implicit — exactly as the count is implicit for
`neg_binomial(mean=…, r=…)`. A likelihood names its _parameters_ (n, p), never
the observed value.

**First guidance — name the block for what is actually observed.** The trap
(caught in review): a block named `prevalence_survey` whose column holds
_positive counts_ reads badly — "prevalence" says proportion, the column holds
integers, and the _honestly_-named column (`tested`) is the one that's secretly
the input. So a survey file `time region
prevalence_survey tested` is genuinely
confusing. The fix is mostly free — **name the block after the observed
quantity**: `slide_positives`, not `prevalence_survey`. Then the file is
`time region slide_positives
tested`, block-name == column == "these are
positive slides," and the input (`tested`) is visibly the odd one out. All
worked examples below follow this.

**The escape hatch — optional `column =` rename.** When the natural block name
still doesn't match the desired column header (e.g. a fixed external schema), an
**optional** `column =` overrides the outcome column:

```camdl
slide_positives[r in region] : {
  projected  = prevalence(I[region = r]) / pop[r]
  schedule   = from_data
  column     = n_positive          # outcome column header is `n_positive` (external schema)
  likelihood = binomial(n = data.tested, p = projected)
}
```

`column =` defaults to the block name, so simple blocks stay terse; it is there
only when an external column name is fixed. It must be honored **symmetrically**
— `simulate` writes the outcome under `column` and `fit` reads it under `column`
— or the §4 round-trip breaks (§8 lists this).

### 7.2.5 Simulate of an exogenous-input block (LOCKED)

At `simulate` dispatch, scan each block; if `needs_design b` and no design is
bound for it, **hard-error early** — not a panic in the sampler, not a silent
NaN (`Data_col` reaching the rmeasure evaluator with no data is otherwise an
index into an empty vector, `obs_model.rs:146+`):

Error language matters for the non-SWE user (review flagged "fit-native /
exogenous" as jargon). Plain version:

```
$ camdl simulate model.camdl --params truth.toml --obs-only synth.tsv
error[E3xx]: block 'slide_positives' needs survey inputs that only exist when fitting:
  · its observation times come from the data file   (schedule = from_data)
  · its denominator comes from the data file        (n = data.tested)
  When you SIMULATE there is no data file, so camdl can't invent these.
  Give it a survey plan — a TSV of when surveys happen and how many were tested:
      camdl simulate ... --plan slide_positives=plan.tsv   (columns: time, region, tested)
  Or, if every survey tests a fixed number, make n a model input (a table):
      tables { survey_n : region = read("...") };  likelihood = binomial(n = survey_n[region], ...)
```

A **plan** (the design file) is a TSV of inputs only — the exogenous columns +
times, no outcome. simulate reads it, draws the outcome, and writes a _complete_
`(outcome, inputs)` dataset that feeds straight back into `fit`. Naming/flag
notes (minor, but decide before P2): use a dedicated `--plan` flag rather than
overloading `--data` (which means "the data to score" in `fit`); and a plan file
is _structurally_ an observation file with the outcome column absent, so the
union/missing loader (§5) can read it as "outcome unobserved → sample it" rather
than inventing a separate format. Both kept out of the LOCKED core.

### 7.2.6 Worked examples (all six — LOCKED surface)

**(1) Weekly counts — common case, fully self-contained.** No `data`, no design;
round-trips perfectly.

```camdl
weekly_cases : {
  projected  = incidence(infection)
  schedule   = every(7 'days)
  likelihood = neg_binomial(mean = rho * projected, r = k)
}
```

simulate draws `weekly_cases ~ NB(rho·projected, k)`; fit scores the
`weekly_cases` column. `simulate --obs-only synth.tsv` → `fit --data synth.tsv`
closes with zero renaming.

**(2) Counts with a time-varying reporting rate** — covariate via the existing
interpolant path (`forcing {}`; the resolve context for a likelihood carries
`time_func_index`, so `reporting(t)` evaluates at the observation time in _both_
directions — verified `obs_model.rs:37`). Still self-contained; no `data`, no
design.

```camdl
# forcing { reporting : interpolated 'ratio { data="reporting.tsv" time_col=t value_col=rho method="linear" } }
weekly_cases : {
  projected  = incidence(infection)
  schedule   = every(7 'days)
  likelihood = neg_binomial(mean = reporting(t) * projected, r = k)
}
```

**(3) Cross-sectional survey — `n` genuinely from data.**

```camdl
prevalence_survey[r in region] : {
  projected  = prevalence(I[region = r]) / pop[r]
  schedule   = from_data
  column     = positive
  likelihood = binomial(n = data.tested, p = projected)
}
```

fit data: `time region positive tested`. simulate needs a design
(`time region tested`) supplying the times (`from_data`) and `tested`
(`data.tested`) — one design covers both, since they are the same exogenous "a
survey happened here, this many tested":

```
camdl simulate model.camdl --params truth.toml --data prevalence_survey=design.tsv
```

**(4) Survey where `n` is a static design — table, NOT `data`.** The decision
rule in action: a static-per-stratum `n` is a table, so the block is
self-contained and round-trips with no design file.

```camdl
tables { survey_n : region = read("design.tsv") }   # static, per-region

prevalence_survey[r in region] : {
  projected  = prevalence(I[region = r]) / pop[r]
  schedule   = every(1 'years)
  column     = positive
  likelihood = binomial(n = survey_n[region], p = projected)
}
```

`needs_design` is false → simulate samples freely. (Verified: a likelihood
resolve context carries `table_index`, so `survey_n[region]` works in a
likelihood today.) This works **only because `n` is time-invariant** — the same
value for every yearly survey in a region. If `n` varied year-to-year (a (time,
region) design), it could _not_ be a table (tables carry no time axis, §7.3) and
would have to ride the data channel as `data.tested` with a design file (ex. 3).

**(5) Age-stratified seroprevalence.** Binomial seropositive-by-age; `n` (number
sampled per age bin) is survey design → `data`. The age axis is a dimension; see
§8.6 for whether the cohort-cumulative projection needs a dedicated `Projection`
variant or rides `DerivedExpr`.

```camdl
seroprevalence[a in age] : {
  projected  = seropositive_fraction[age = a]    # model's predicted P(seropos | age)
  schedule   = from_data
  column     = seropositive
  likelihood = binomial(n = data.sampled, p = projected)
}
```

**(6) The illegal state — now unrepresentable.**

```camdl
# Loose-field surface allowed this; the parser had to reject it at runtime:
#   every = 7 'days
#   at    = [90 'days]      ← two schedules
# Option B: `schedule =` takes exactly one variant. You cannot write it.
```

### 7.3 The table-vs-data decision rule (for the model author)

The deciding axis is **what the quantity is indexed by** — _not_ whether you can
predict it. (Tables cannot carry a time axis — `TableLookup` resolves to an
integer flat index, `resolved_expr.rs:59`, `propensity.rs:189`; the only
continuous-time path is the interpolant forcing. So "predictable" is the wrong
axis: a _known but time-varying_ design — "test 300 in 2015, 250 in 2016" — is
perfectly predictable yet **cannot** be a table, because it is indexed by (time,
stratum).)

| The quantity depends on…                                                                                                    | Goes in                                      | Why                                                                                                                |
| --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **continuous time** (reporting rate, coverage)                                                                              | `forcing { interpolated }` (→ rename gh#138) | only continuous-`t` path; evaluable at any obs time                                                                |
| **discrete stratum only, time-invariant** (per-region reporting prob; an `n` that is the same for every survey in a region) | `tables {}`                                  | static, dimension-indexed; **round-trips**                                                                         |
| **the observation event** — varies by time / survey-to-survey                                                               | the **data channel** via `data.col`          | time-indexed like observation data; a _design file_ supplies it when simulating, the real survey file when fitting |

The crucial subtlety this rule encodes: a **time-varying `n` design is not a
`tables{}` table** — it rides the _same data channel_ as the real data, because
the design file (`time region tested`) is time-indexed exactly like observation
data is. Predictability only decides _which file you point at_ (a design you
authored vs the real survey), never the mechanism.

One-sentence rule for the author: **continuous-time covariates → forcing;
time-invariant per-stratum constants → tables; anything that varies with the
observation event (the survey's time, its turnout) → the data channel
(`data.col`), with a design file standing in for the real data when you
simulate.** Litmus for `n`: _same for every survey in a stratum → table; varies
survey-to-survey or over time → data channel_ — because tables can't carry time,
anything time-varying (even a known design) must ride the data channel.

---

## 8. IR schema changes

Per "Changing the IR schema" in CLAUDE.md (atomic OCaml+Rust+golden):

Each item below must be applied atomically across **all** of: `ir/schema.json`,
OCaml (`ocaml/lib/ir/ir.ml`, `serde.ml`, deserialize), Rust
(`rust/crates/ir/src/`), then `make test-unit` and
`make update-golden && make update-expected` (CLAUDE.md 6-step).

1. **`ObservationModel`** gains an optional `aux_columns: [String]` (the data
   columns a block's likelihood references via `data.<col>`), so the loader
   knows which extra columns to retain. **Derived, not declared:** collected
   from the same expression walk that resolves the likelihood (§7.2.2) — never a
   hand-maintained list. _(OCaml ir.ml + serde + Rust + schema.)_
2. **`ObservationModel`** gains a `window_semantics` enum (`fixed_cadence`
   default | `cumulative_since_last_report`, §5.2.1) — load-bearing for
   incidence + missing data. _(OCaml + Rust + schema.)_
3. **New `Expr` variant `DataCol(name)`** (`Data_col` in OCaml). **Blast radius
   is large — ~18 match sites, NOT the ~5 an earlier draft implied.** Every
   full-`Expr` match in the codebase is exhaustive with no wildcard, so each is
   a hard compile error until a `DataCol` arm is added (verified by the
   engineering review). Required edit sites:
   - _Rust:_ `expr.rs` (enum **+ the manual `impl Deserialize for Expr`** match
     arm — Serialize is derived but Deserialize is hand-written);
     `resolved_expr.rs` **×4** (`resolve_expr`, `eval_resolved`,
     `eval_resolved_deriv` [PGAS gradient — omit and gradients silently
     mis-handle], `references_state` → returns **false**); `propensity.rs`
     **×3** (`eval_expr` slow path, `eval_expr_deriv`, and the
     time-dependence/param-ref scanners `expr_is_time_dependent` /
     `expr_refs_param` — a `data.col` arg is _per-observation-varying_, so wrong
     arms here silently mis-cache); `ir/validate.rs::check_expr` — **reuse the
     existing `allow_projected: bool` flag pattern** for the scope restriction
     (rule 4, §7.2.2), do **not** invent a new E-code path; `pgas.rs` and
     `hierarchical.rs` gradient/match sites.
   - _OCaml:_ `ir.ml` (AST), `serde.ml` (`expr_to_json` **and** `expr_of_json` —
     the latter fails closed on unknown keys), `dimcheck.ml` `infer` arm (return
     unknown-dim via `fresh_var`; reconcile with the **existing**
     permissive-likelihood mode rather than a parallel mechanism),
     `validate.ml`, `autodiff.ml`, `pp_expr.ml`, `expander.ml`.
   - Resolved at `MultiStreamObsModel::new`; rmeasure sampler errors per §7.2.5
     (the sampler's `EvalCtx` has **no data channel today** — a per-observation
     input slot must be threaded, §7.2.5 plumbing not yet budgeted). Correct the
     §7.2 "off the hot path" wording: the _resolution_ is once at construction;
     the _eval_ (`eval_resolved`) **is** the hot path — a resolved `DataCol` is
     an O(1) array index there, which is the actual requirement to state.
4. **`ObservationModel`** gains an optional `column: Option<String>`
   (outcome-column rename, §7.2.4; default = block name). Must be honored on
   **both** the simulate-write path (`main.rs:1146` writes `name` today) and the
   fit-read path (`load_data_tsv_column`), or the §4 round-trip breaks for
   renamed blocks. _(OCaml + Rust + schema.)_
5. **DELETE the vestigial `data_stream` field.** `ObservationModel` has both
   `name` and `data_stream` (`observation.rs:85`), but `data_stream` is read
   **nowhere** in Rust for binding/scoring (verified: the only non-test
   reference is its own field definition; every binding path uses `name` —
   `pfilter.rs:206`, `runner.rs:285`, simulate `main.rs:1079`). It is the exact
   dead field that misled a reviewer into a phantom "two name fields" finding.
   Per "delete dead code on sight," remove it from Rust `observation.rs`, OCaml
   `ir.ml` + serde + the expander's `odata_stream`/W203 path
   (`expander.ml:3782`, `:3997`), and `schema.json`; regen golden. The
   stratified-sharing job it nominally did is now covered by long-format + the
   `name`-binding convention. **(This is the first, standalone deletion commit —
   landed before the rest, since it's pure cleanup and shrinks the surface every
   later change touches.)**
6. **`ObservationSchedule` surface — a real grammar change, NOT cosmetic.** The
   _IR_ enum is already 3-variant `AtTimes|Regular|
   FromData` (no IR change).
   But the **DSL surface has no `from_data` at all today**: the surface AST is
   2-variant (`ObsEvery|ObsTimes`, `ast.ml:161`), the parser uses loose
   `every=`/`at=` fields (`parser.mly:480`), there is no `from_data` token in
   the lexer, and the expander has no `ObsFromData` lowering. So
   `schedule =
   every(..)|at(..)|from_data` (§7.2.3) requires: a 3-variant
   surface AST, a `from_data` keyword (LR-conflict check against the 10 existing
   `EVERY`/`AT` uses), a rewritten `obs_kv` rule, and the `ObsFromData`
   lowering. Also: `from_data` is currently a **warn-and-fabricate stub in
   simulate** (`main.rs:1352` prints a warning and invents a unit grid) — the
   `needs_design` hard-error (§7.2.5) must _replace_ that stub. Verify
   `FromData` covers irregular times.
7. **Conditioning boundary** — a `t_cond: Option<f64>` on the simulation/fit
   config (§6). OCaml serde + Rust + schema. The unweighted-ensemble warm-up
   logic lives in the Rust filter, not the IR.
8. **Per-stream flow-accumulator reset (§5.2.2).** Not a schema change but a
   `ParticleState.flow_accumulators` + `particle_filter.rs` (and PGAS/IF2
   mirror) change: make `reset_flows` per-accumulator, indexed by which stream's
   cadence owns the flow. Load-bearing for multi-cadence incidence fitting;
   lands in P1.5.
9. **Remove the three identical-times gates (§5.1 union axis).** All _three_
   must go atomically or whichever is missed silently re-rejects ragged input:
   `multi_stream_obs.rs:299-306`, `runner.rs:296-308`, `pfilter.rs:166-179`. (An
   earlier draft cited only two.)
10. **Serocatalytic projection — no engine change (§7.1, A1).** Resolved:
    age-axis seroprevalence rides `DerivedExpr`/`CurrentPop` over an
    age-structured _model_; it needs a worked age-structured model
    (doc/example), **not** a new `Projection` variant. No schema change; the
    deliverable is the example, scoped to the sero follow-up, not this proposal.
11. Bump `ir/VERSION`; regenerate all golden + expected files.

---

## 9. CAS materialization of the joined table

The materialized union table **is** the joined data: write
`joined_observations.tsv` (long layout — faithfully represents ragged data,
shows exactly which `(time, stratum, stream)` cells were present, skipped, or
structural-zero) plus `joined_observations.json` (typed manifest: per-stream
kind, missing-count, time range, source file hashes, join mode). Benefits:

- the modeler eyeballs exactly what aligned to what and where the NaNs landed
  before burning compute;
- the fit's content hash now covers the _resolved_ data, so a re-run with a
  silently-changed input file is detectable;
- the missing-vs-zero and cadence decisions become inspectable artifacts, not
  buried runtime behavior.

---

## 10. The block↔column binding interface

Five forms, in order of how much they help; keep the simple ones, recommend the
auditable one:

1. **`--data block=file.tsv`** (current) — column = block name, 2-col fallback
   to column 1. Keep as the quick path.
2. **Wide auto-bind** — one TSV `time, cases, deaths`; each column to its
   like-named block. Strict on extra/missing columns (error + list).
3. **Tidy/long `time, <dim>, value`** — the spatial/stratified unlock; maps
   directly onto indexed observation blocks. _New._
4. **Column override** — `--data cases=file.tsv:notif_count` to break the name
   coincidence when a column is not named like its block.
5. **Declared manifest** (`[data.observations]` extended) — per stream:
   `block, file, column(s), time_column, missing sentinel, format`. The
   recommended, reproducible path; `--data NAME=PATH` desugars into it.

Reconcile fit-vs-pfilter subset strictness: today fit errors on zero streams
(`runner.rs:258`) but pfilter only _warns_ on a partially-bound multi-stream
model. Make both error by default on an unbound block, with an explicit
`--allow-unbound` opt-in.

---

## 11. Type sketch (consolidated)

```
ObservationBlock = {              # what the model author writes
  name        : String            # value-column header; output column name
  index       : [Dimension]       # [] scalar; [region] for cases[r in region]
  kind        : Projection        # incidence | prevalence | sero — the `projected=` expr
  schedule    : Every(dur) | At([t]) | FromData
  likelihood  : Likelihood        # args are model exprs OR data.<col> bindings (new)
  aux_columns : [String]          # data columns referenced via data.<col>
}

DataBinding = {                   # how a file binds to blocks (= tables §6)
  layout      : Wide | Long
  index_cols  : [time, <dim>…]    # matched BY NAME to dimension levels (§6.3)
  value_cols  : [<blockname>…]    # matched BY NAME to blocks (round-trip contract)
  missing     : RowAbsent | NaNCell    # both = skip under MAR
}

Join = union (DEFAULT; union axis, absent (time,stream) = unobserved) | strict (require identical axes)
```

---

## 12. What this is _not_

- Not a tables-format change (tables stay positional; observations are by-name —
  §2).
- Not backwards-compatible shims — alpha; golden files updated atomically.
- Not MOI/molecular-FOI equilibrium likelihoods (deferred).
- Not a continuous-Normal PDF likelihood (separate request; current `normal` is
  discretized-count).

---

## 13. Implementation phases (risk-ordered)

The low-risk format/loader work ships first; the inference-math change ships
last behind it.

The original draft bundled the low-risk reader work with the high-risk
missing-skip-at-the-inference-seam work into one "P1." The adversarial review
(§15) showed missing-skip for incidence carries a window-semantics correctness
trap (§5.2.1) as subtle as the `t_cond` work. **Split them.**

- **P0 — Delete the vestigial `data_stream` field (§8 item 5).** Pure cleanup,
  no behaviour change; it's the dead field that misled a reviewer, and it
  shrinks the `ObservationModel` surface every later phase touches. Standalone
  first commit.
- **P1 — Long/wide reader + format only.** Reader auto-detects layout (long via
  dimension-named index column; wide via block-named value columns); long output
  for stratified blocks (§4, includes the golden regen + the
  `cases_r0`-demangling ambiguity hard-error); manifest/CAS plumbing of the
  _format_. **No inference-seam change.** Lowest-risk; lands early.
- **P1.5 — `union` axis + missing-skip + window semantics + guardrail. THE
  correctness tier.** Remove **all three** identical-times gates atomically (§8
  item 8); union join (default); per-point skip at the `multi_stream_obs.rs`
  seam with the "weight-skip ≠ step-skip" rule (§5.2, all-absent → no
  weight/resample but still stop to reset); **score ≠ reset** + **per-stream
  accumulator reset** (§5.2.1 + §5.2.2 — the multi-cadence incidence lift, the
  largest item here); the first-window / out-of-family-stream guardrail (error +
  opt-out, §6, §14 Q1); watchdog-budget scaling for widened intervals (§15).
  Inference-math; heaviest test load (golden + property + a negative control per
  avoid-vacuous-tests, incl. a test that an off-grid second stream does NOT
  truncate the first stream's incidence window). **This is what actually makes
  the joint malaria fit (cases + survey) possible** — see P2.
- **P2 — `data.<col>` + survey/cross-sectional blocks.** New `Expr` variant
  (`Data_col`, ~18 match sites, §8 item 3; scope-restricted via the existing
  `allow_projected` flag pattern), derived aux-column loading, optional
  `column =` rename (§7.2.4, symmetric read/write), Option-B `schedule =`
  surface (the real grammar change, §8 item 6), `needs_design` simulate guard +
  plan-file path (§7.2.5), **`observed ≤ n` validation (§7.2.1)**, dimcheck
  unknown-dim. **Dependency (was unstated): P2 surveys fit _jointly_ with case
  series only after P1.5's union axis + per-stream reset.** On its own, P2
  delivers _single-stream_ surveys; the headline joint multi-cadence malaria fit
  needs P1.5 **and** P2. Resolve §14 Q4 (plan-file vs `n=`-expression) before
  starting — it gates the IR validation here.
- **P3 — Tidy-long ingestion ergonomics + CAS materialization + manifest
  interface.** The UX/auditability layer. (CAS materialization can't precede
  P1.5 — there's no joined table until the union axis exists.)
- **Deferred (pinned) — `t_cond` + burn-in (the measles _fix_).** Split to
  [`2026-05-30-conditioning-boundary-tcond.md`](2026-05-30-conditioning-boundary-tcond.md);
  revisit after P1–P3. The §6 guardrail (in P1.5) is the interim safety net that
  makes measles _safe_ (loud error on the pathological window) without _solving_
  the burn-in.
- **Deferred — spatial aggregation** (§7.1; model-strata-but-score-the- sum,
  needed for tractable national spatial fits per Snyder et al.),
  **age-structured seroprevalence model** (§7.1; a worked example, not engine
  work), **EIR/continuous-log likelihood family** (§7.1), and **MOI/molecular
  FOI**.

Each phase: red→green TDD, golden regen, no lowered gates.

---

## 14. Open questions

**Resolved 2026-05-30 (Vince):**

1. ~~Guardrail severity~~ → **error + opt-out** (§6, §14 Q1; P1.5).
2. ~~Scope: keep or split `t_cond`~~ → **split out and pinned**
   ([`2026-05-30-conditioning-boundary-tcond.md`](2026-05-30-conditioning-boundary-tcond.md)).
3. ~~`data.<col>` vs `denominator =`~~ → **`data.<col>` as a reserved,
   runtime-only namespace** (§7.2.2, four rules). Outcome column is the block
   name with an **optional `column =` rename** (§7.2.4), not a mandatory
   `outcome =` field. Schedule is a single tagged
   `schedule = every(..)|at(..)|from_data` (Option B, §7.2.3), not loose fields.
   Reporting completeness is **not** `data` — it's a covariate via the
   interpolant forcing path (§7.3 decision table). All LOCKED.
4. ~~`union` join default~~ → **`union` by default** (include all data; safest =
   never silently drops; §5.1), with the out-of-family-stream guardrail against
   the typo failure mode.

**Still open:** 4. **Survey round-trip denominator source** (§4 wrinkle,
§7.2.5): for the _sampling_ (simulate) direction of a `data.<col>` survey block
— a `--data block=design.tsv` design file (LOCKED as the primary path, §7.2.5),
with a static-`n` **table** as the round-trip-free alternative (§7.2.6 ex.4).
Open sub-question: should an `n =` model-expression also be accepted as a third
form, or do design-file + table cover it? (Determines §8 IR validation: a
`DataCol`/`from_data` block must have a design bound or `simulate` hard-errors
per §7.2.5 / `needs_design`.)

---

## 15. Adversarial review log (2026-05-30)

A subagent review verified every file:line claim against the code and attacked
the design. Outcome: core sound, two **critical** holes found and fixed in this
revision.

- **CRITICAL — `t_cond` warm-up incoherence** (was §6). Original "ODE
  deterministic skeleton" is incoherent for chain_binomial integer models and
  would seed a variance-free cloud. **Fixed:** rewrote §6 as an _unweighted
  stochastic ensemble_ warm-up (correct process variance; reuses existing
  propagation; lower risk).
- **CRITICAL — incidence + missing-data window semantics** (was §5.2). MAR ≠
  window-correctness; skipping a likelihood without resetting the accumulator
  re-creates gh#134. **Fixed:** added §5.2.1 (score ≠ reset; grid-tied reset;
  per-block `fixed_cadence` / `cumulative_since_last_report`).
- **MAJOR — stratified round-trip false** (§4). simulate emits `cases_r0` wide;
  long form is `region,cases`. **Fixed:** §4 now scopes the round-trip guarantee
  to long format for stratified blocks
  - requires long simulate output.
- **MAJOR — no `observed ≤ n` survey guard.** **Fixed:** §7.2.1.
- **MAJOR — `data.<col>` sampling-direction panic.** **Fixed:** §7.2.5
  (`needs_design` predicate + dispatch-time hard error; design-file path).
- **MAJOR — watchdog interaction on all-absent / warm-up stretches** (review
  DEFECT 9, upgraded from the draft's MINOR). A folded all-absent stretch or the
  §6 warm-up is one long propagation with no resample, which collides with the
  _existing_ gh#133 degeneracy watchdog two ways: (a) the wall-clock budget
  (`degeneracy.rs:67`, `WALLCLOCK_PER_PARTICLE_S`) scales with particle count
  but **not** with inter-observation interval length, so a long fold can
  false-trip `PFWallclockTimeout`; (b) the ESS-collapse window count
  (`ESS_COLLAPSE_WINDOWS = 3`) is per-obs-step, so folded steps that push no ESS
  entry desync "3 consecutive windows" from wall-clock time. This is a **P1.5**
  concern (the fold lands in P1.5), not a new P4 guardrail — an existing
  watchdog already fires, possibly with the wrong error _class_ (resource-limit
  reported as statistical degeneracy). §5.2/§6 must specify: pause the
  wall-clock timer during warm-up, decide whether folded steps push ESS entries,
  and rescale the per-window budget for widened intervals. This reframes §14 Q1:
  the guardrail is partly _already built_ and mis-firing, not absent.
- **MINOR — schema list incomplete; sero projection gap.** **Fixed:** §8 (OCaml
  serde items + sero decision §8.6). NOTE the review found §8 still understates
  the `Expr::DataCol` blast radius: it needs the **manual
  `impl Deserialize for Expr`** match arm (`expr.rs:319-357`, the documented
  footgun at `expr.rs:278-286`), a **`ResolvedExpr` mirror + `eval_resolved` +
  `references_state` arm** (`resolved_expr.rs` — this _is_ the hot path, contra
  §7.2's "off the hot path"), a **`dimcheck.ml` `infer` arm** returning
  unknown-dim, and the **`schema.json` `observation_model` aux_columns
  property**. Fold all into §8 before implementation; the one-line "in both
  expr.rs and OCaml" understates it.
- **MAJOR — `data_stream` IR field vs `name` binding** (review DEFECT 6).
  `ObservationModel` has _two_ string fields, `name` and `data_stream`
  (`observation.rs:84-85`, both required at `schema.json:646`), but every
  data-binding path binds by `name` (`simulate` writes `o.name` at
  `main.rs:1079`; loaders find by `o.name` at `pfilter.rs:206`,
  `runner.rs:285`). So `data_stream` is **vestigial w.r.t. the data path** this
  proposal touches — yet the column-override case (§10 form 4,
  `--data block=file.tsv:col`) is arguably exactly what `data_stream` was meant
  to express. **Decide:** either delete `data_stream` (per "delete dead code on
  sight") or repurpose it as the canonical column-override slot. Do not leave a
  field in the IR contract that _looks_ like it names the data column but is
  ignored — that is a trap for the implementer. Schema change either way.
- **Reinvention — `at = data` duplicated `FromData`.** **Fixed:** §7.2 now
  reuses `FromData` via `schedule = from_data`.

Line-number drift noted by the reviewer (cosmetic, corrected in text):
heterogeneous-reject is `multi_stream_obs.rs:308` (not :300); flow reset is
`particle_filter.rs:377` (not :364).

(The first-round bullets above use earlier section numbers; some have drifted as
sections were added — read them as history, not a live index. The §8.6
references mean today's §8.9, serocatalytic.)

### Second review round (2026-05-30, four-lens panel) — folded in

Four adversarial reviewers (engineering / abstraction / researcher-UX / scope)
re-read the revised proposal. Net: design sound, no redesign; corrections were
_scope-honesty_ and two correctness specs. Folded in:

- **B1 — self-contradiction (CRITICAL, was in §5.2/§5.2.1).** "All streams
  absent → fold `t_k` into the next interval" contradicted "reset at every grid
  point." **Fixed §5.2:** weight-skip ≠ step-skip — still stop at the checkpoint
  to reset, just don't weight/resample.
- **F1 — multi-cadence per-stream reset (MAJOR, the real lift).** The global
  `reset_flows` (the `particle_filter.rs:364` canary) breaks when a second
  stream's off-grid timestamp resets the first stream's incidence accumulator.
  **Added §5.2.2** + §8 item 7 + P1.5 scope: reset becomes per-flow, indexed by
  owning cadence. This is what actually gates the joint malaria fit.
- **`Data_col` blast radius (CRITICAL under-scope).** ~18 exhaustive match sites
  across both languages, not ~5; reuse the existing `allow_projected` flag for
  scope-restriction. **Rewrote §8 item 3.**
- **`from_data` has no DSL surface today (MAJOR mis-scope).** The "surface-only,
  easy" schedule change is a real LR-grammar edit + a warn-and-fabricate
  simulate stub to replace. **Rewrote §8 item 6.**
- **Scope honesty (MAJOR — proposal oversold).** Measles is _diagnosed +
  guarded_, fix deferred to pinned `t_cond` (§0). Seroprevalence = a
  modeling-pattern follow-up, not delivered; EIR needs a new likelihood family;
  spatial aggregation cited-as-necessary but undesigned — all named deferred
  (§7.1). P2 surveys need P1.5 to fit _jointly_ (§13).
- **MNAR honesty (MAJOR).** Completeness covariate fixes reporting-rate
  variation, NOT outcome-dependent (stockout) missingness — latter named out of
  scope (§5.2.2).
- **`data_stream` deletion (the field that misled round 1).** Confirmed read
  nowhere for binding; **delete it** as P0 (§8 item 5, §13).
- **UX (researcher lens).** Block-name == column trap fixed by naming blocks for
  the observed quantity (`slide_positives`, §7.2.4); plainer simulate error,
  `--plan` not overloaded `--data` (§7.2.5); a mandatory pre-fit
  schema/missing-data echo recommended as the single highest-leverage
  anti-silent-failure mitigation (P1, noted).
- **Kept as-is (deliberately):** the `data` namespace name (clear enough with
  the prefix); NaN-as-missing sentinel (§0, good enough for now); the
  three-channel forcing/table/data decision (principled — rejected the "give
  tables a time axis" merge, which would just make a table a forcing/data in
  disguise).

## 16. References

- gh#134 (calendar-time ergonomics); the Kano measles repro (§1.1).
- pomp FAQ §3.3 "How do I deal with missing data?" (NA-row convention; verbatim
  text to confirm).
- He et al. (2010) _J. R. Soc. Interface_ 7:271–283 — book vignette
  `he2010_london.camdl` (§6 comparison).
- Snyder, Bengtsson, Bickel & Anderson (2008) _Mon. Wea. Rev._ — curse of
  dimensionality in particle filtering (motivates aggregation, user's note).
- Malaria/sero/surveillance sources cited inline in §5, §7.
