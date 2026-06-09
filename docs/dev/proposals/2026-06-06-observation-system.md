---
date: 2026-06-06
status: proposal
supersedes: 2026-06-05-observation-data-binding.md
related:
  - 2026-06-06-scheduling-effect-topology.md
  - 2026-05-14-reactive-interventions-and-evsi.md
area: observation data loading / inference
issue: gh#171, gh#172, gh#98, gh#134
---

# Observation system

## Scope: the data layer, on top of the timeline spine

The observation surface splits into two layers that were previously entangled,
and the split is what lets this land cleanly:

- **The temporal layer** — reconciling observation _times_ with integrator steps
  — is owned by the
  [scheduling-effect topology](2026-06-06-scheduling-effect-topology.md). That
  work ships the `Observe` effect (a read-only `&State` projection at an obs
  boundary), first-class `TemporalKind { Interval, Instant }`, the per-stream
  `ResetWindow{ flow_indices }` (the `Stage::Reset` accumulator close), the
  `StepPolicy { Snap, Exact }` off-grid reconciliation, and the runtime sub-`dt`
  collision guard. Anything that is a per-substep state read/write or a
  time→step mapping lives there.
- **The data layer — this proposal** — turns untyped data rows into a
  model-shaped, fully-typed object, and routes every fitting algorithm to
  consume _one_ such object instead of each re-loading and re-checking `--data`.
  Anything that is _parsing and validating rows into model cells_ lives here,
  and it touches no substep.

The two meet at exactly one mechanical mapping (below): each bound stream
becomes one `Observe` effect, and each interval stream contributes one
`ResetWindow`. The data layer is independent of the timeline and can proceed in
parallel with the topology implementation; only the final union-axis scoring
change waits on the per-stream `ResetWindow` the topology work ships.

## Framing: bind, not join

Loading observation data is not a symmetric **join** (two co-equal relations,
the result the union of both schemas, neither side privileged). It is an
asymmetric, directional **bind**: the model _defines_ a fixed lattice of named
cells — which streams exist, which strata, which times — and data values are
bound _into_ those named slots, like binding arguments to parameters. The
framing privileges the model's lattice as the authority, and that changes the
semantics in exactly the way a correctness surface needs:

- a **leftover** (a data row with no cell — a typo'd stream, a stratum the model
  lacks, an off-grid time) is "data I have nowhere to put" → usually a mistake;
- a **hole** (a cell with no data) is "this slot got nothing" → often _expected_
  (sparse surveillance).

A symmetric join collapses both into one "unmatched" bucket; a bind keeps the
two directions distinct, with distinct severities. And the result type differs:
a join yields a union of two tables; a bind yields a **model-shaped object with
typed holes** (`Option` cells) — the sparse-geometry representation `gh#171`
needs.

## Why this is a correctness surface

A data point that fails to reach scoring — silently — is a wrong likelihood,
hence a wrong posterior. The bind exists to make every malformed input a _named,
located_ outcome. Three residual gaps motivate it:

- **NaN/inf are unguarded on the obs value path.** `pfilter.rs:669,699` parse
  values with bare `parse::<f64>()`, which accepts `"NaN"`/`"inf"`; nothing
  checks `is_finite` before the value reaches the log-pmf.
- **Loading is scattered and duplicated.** The
  `--data → per-stream series → MultiStreamObsModel` pipeline is re-implemented
  in `pfilter.rs`, `profile.rs`, `fit/runner.rs`, and `survey.rs` — each
  resolves `--data`, builds its own per-stream series, replicates the
  shared-grid check (≥5 sites), and canonicalizes
  `observations = per_stream_obs[0]`. That duplication is where silent drops and
  the homogeneity asserts live.
- **Holes cannot be expressed.** Every stream shares one dense `obs_times`, so
  sparse/multi-cadence surveillance (`gh#171`) is rejected, not represented.

(The sub-`dt` collision drop in `build_obs_at_substep` — two obs rounding to one
substep, last-wins — is a _temporal_ hazard the topology work's runtime
collision guard closes; the data layer's `Collision` finding catches the
load-time version.)

## The data-layer types (`obsdata`)

Data enters as untyped rows and leaves as a value the inference traits already
consume. Nothing downstream is new; the work is to give the existing scoring
seam a single, typed input.

```rust
mod obsdata {
    // ── input: one untyped row per PRESENT observation ──
    struct LongRow { stream: String, stratum: Option<String>, when: RawTime, value: RawValue }
    enum RawTime  { Offset(f64), Date(String) }      // resolved via ir::caltime + model origin
    enum RawValue { Num(f64), Missing, Unparseable(String) }

    // ── output: a model-shaped, fully-typed object ──
    /// PRIVATE ctor — only `bind` constructs one, so no un-validated data can reach
    /// the likelihood. Every leftover/collision/hole is accounted for in a
    /// BindReport before any value reaches scoring.
    pub struct BoundObs { times: Vec<f64>, streams: Vec<StreamCells> }
    struct StreamCells {
        name:  String,
        kind:  TemporalKind,            // the SAME type the runtime carries on Observe
        cells: Vec<Option<ObsCell>>,    // None = hole; one slot per time in `times`
    }
    /// `Scalar` is the common case; `Counted` carries a per-observation denominator
    /// — a Binomial/BetaBinomial n that varies survey-to-survey (the malaria case).
    enum ObsCell { Scalar(f64), Counted { value: f64, denom: f64 } }

    // ── the report: errors are VALUES, not control flow ──
    pub enum Severity { Error, Warn, Info }
    pub struct Finding { kind: BindIssue, stream: String, detail: String, count: usize, severity: Severity }
    pub enum BindIssue {
        LeftoverColumn, LeftoverStratum, OffGridInterval, OffGridInstant,
        Collision, Duplicate, CoarserThanModel, Hole, RejectedValue,
        UnparseableDate, InconsistentTimeColumn,
    }
    pub struct BindReport { findings: Vec<Finding>, verdict: Severity }

    pub fn bind(model: &Model, rows: Vec<LongRow>, dt: f64, cal: &CalendarCtx, policy: &BindPolicy)
        -> (BoundObs, BindReport);   // never panics, never exits — errors are VALUES (gh#181)
}
```

`TemporalKind` is **not** declared here — it is imported from the topology work,
where it is a runtime type carried on the `Observe` effect. The loader _chooses_
the variant from the stream's projection; the runtime _enforces_ its semantics.
One type, two responsibilities, no duplicate definition.

## The bind as a cardinality map

`bind` is a partial map `φ : DataRow → ModelCell`, cell =
`(stream, stratum, k)`. Every failure is a departure from "injective and total,"
with a defined resolution and a severity that splits by _direction_:

| cardinality            | cause                                        | resolution                                                                                     |
| ---------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 1:1                    | —                                            | clean                                                                                          |
| many:1 (non-injective) | `dt` coarser than data, or duplicate row     | `Collision`/`Duplicate` → **Error**; `--aggregate=sum\|mean` opt-in (loud, changes likelihood) |
| 1:many                 | data coarser than model (region vs district) | needs a model aggregate cell (`CumulativeFlowSum`); else `CoarserThanModel` → **Error**        |
| 0:1                    | cell with no data — a hole                   | `None` cell; `Hole` → **Info** (sparse) / **Warn** (stream declared dense)                     |
| 1:0                    | data with no cell — a leftover               | `LeftoverColumn`/`LeftoverStratum` → **Error**; benign metadata column → **Info**              |

Data-has-extra (`LeftoverColumn`) defaults to Info (real files carry
`population`, `notes`); model-cell-unfilled-when-dense and stratum-mismatch
default to Error.

## Binding and typing columns

A loader has to answer three separate questions about every file — which column
is which, is the time column trustworthy, is the value cell trustworthy — and
the "bind, not join" framing gives each a definite, non-guessing answer. These
are where silent mis-loads live, so it is worth spelling out the _why_ of each.

### Column role is bound by name, never sniffed from content

A column's _role_ — time axis, stream, stratum, or value — is decided by
matching the model's declared names, not by inspecting what the cells look like.
This is the bind principle applied one level down, to columns: the model is the
schema, and the file is bound _into_ it.

- The **time** column is the one named `time` (or named by `--time-col`), not
  "whichever column looks date-shaped." A model with a stream literally called
  `t` must not have its case counts mistaken for timestamps because they happen
  to be small integers.
- A header that matches a model **stream** name _is_ that stream; a header that
  matches a **stratum** dimension _is_ that stratum. (The existing column-named
  multi-stream loader already works this way — every column must be named,
  nothing is positional.)
- A header that matches **nothing** is a located `LeftoverColumn` finding —
  surfaced with the column name and a hint listing the model's known stream and
  stratum names. It is `Info` if it looks like benign metadata (`population`,
  `notes`, `source`) and an `Error` otherwise, but it is _never_ silently
  dropped and _never_ content-sniffed into a role.

The failure mode this rules out is the dangerous one: a content heuristic that
silently routes the wrong column into the likelihood. With binding-by-name, "the
loader couldn't tell what this column was" has a definite outcome — it says so,
names the column, and lists what it expected — rather than guessing and
producing a quietly-wrong posterior. There is no code path in which a
mis-identified column reaches scoring.

### Time-cell typing is reused, not reinvented — and has one known gap

Once the time column is _identified_, typing its cells is a solved problem we
reuse rather than rebuild. `caltime_load::convert_time_column` / `detect_kind`
(`caltime_load.rs:100-221`) scans the whole column and decides its kind once:

- all cells numeric → day-offsets (byte-identical to the pre-date behaviour);
- all cells ISO-date → dated, converted through the model's `origin` +
  `time_unit`;
- a **mixed** column → a hard error that names _both_ offending rows (tested:
  `mixed_column_errors_naming_both_rows`);
- a `--time-format numeric|date` override is honoured _before_ detection, with a
  helpful message when `numeric` meets a date cell;
- a dated column with no model `origin` → a hard error with the fix hint
  (tested: `dated_without_origin_errors`).

So "a date-like column with one bad date in the middle" is already a _located_
hard error, not a silent coercion. Under the bind these become `Finding`s
(`UnparseableDate` / `InconsistentTimeColumn`, `Error` severity) so they flow
through the `BindReport` alongside everything else — we are wrapping the
existing detector, not rewriting it.

The one honest gap: the **invalid-but-date-shaped** cell — `2024-13-45` (month
13), or `2024-O3-01` with a letter O for a zero. It hits the right "neither a
number nor an ISO date" branch and produces the located message, but there is
**no test pinning that branch** today (the suite covers the numeric/date _mix_,
not the neither-case). That is a test obligation (below), not a missing
mechanism — but it is exactly the kind of edge a real surveillance export hits,
so we name it rather than assume it away.

### Value-cell typing is the hardening this proposal actually adds

`caltime_load` is time-only; the observation _value_ cells are, today, parsed
with a bare `parse::<f64>()` (`pfilter.rs:669,699`) that silently accepts
`"NaN"` and `"inf"` and has no located error for a non-numeric value. This is
the real robustness hole, and it is on the value path, not the date path. The
`RawValue { Num | Missing | Unparseable }` type plus a finiteness guard closes
it: a `NaN`/`inf` can never reach the log-pmf (where it would silently poison
the likelihood — a `NaN` weight propagates through resampling and an `inf`
collapses the normalization), and a non-numeric value (`"NA"`, `"*"`, a stray
quote, a censoring mark like `"<5"`) becomes a located `RejectedValue` finding
reported with the cell and a count, not coerced. That last example — the
censoring mark — is the seam to the missing-data question, which deserves its
own treatment.

## How `BoundObs` binds to the timeline spine

`BoundObs` is the input to types that already exist, plus the new `Observe`/
`ResetWindow` effects the topology work introduces. The mapping is mechanical,
and it is the single place the data layer and the temporal layer connect:

| `BoundObs`               | becomes, in the runtime                                                                                                    |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `times` (the union axis) | `Schedule::with_obs(times)` — the obs boundaries every driver steps to (Snap or Exact)                                     |
| each `StreamCells`       | one `Observe` effect (`Stage::Observe`, read-only `&State`)                                                                |
| `StreamCells.kind`       | the `Observe`'s `TemporalKind` → its `StreamProjection` (Interval→`FlowSum`; Instant→`IntCompSum`/`Expr`)                  |
| each `Interval` stream   | one `ResetWindow{ flow_indices }` (`Stage::Reset`) keyed to _that_ stream's flows                                          |
| `cells[k] = Some(v)`     | a scored observation for that stream at `obs_idx = k`                                                                      |
| `cells[k] = None` (hole) | that stream contributes **no term** to the joint log-likelihood at `obs_idx = k` — skipped, not scored as an observed zero |

That last row is the entire correctness point: a hole is the _absence of a term
in the sum_, not an observed value of zero. The homogeneous path cannot express
it because every stream shares one dense `obs_times`; `Option` cells over a
union axis can. And the per-stream `ResetWindow` is what makes a _sparse_
interval stream correct: its flow accumulator is zeroed only at _its own_ obs
times, so a weekly-cases observation no longer truncates a monthly-deaths window
(the global-reset corruption the topology work's `M3` fix removes).

The flow, end to end:

```
--data PATH  /  --data NAME=PATH                 (raw file: long, or wide-sugar)
      |  parse + ir::caltime  (date -> model-time)
      v
Vec<LongRow>                                     (untyped: stream, stratum, when, value)
      |  obsdata::bind(model, rows, dt, cal, policy)
      v
(BoundObs, BindReport) -------------------------> report.verdict
      |  model-shaped, typed, Option cells           Error  -> refuse: SimError::Validation
      |                                               Warn/Info -> proceed, surface findings
      v
MultiStreamObsModel : ObservationModel<ParticleState>
 + one Observe effect per stream  (TemporalKind, StreamProjection)
 + one ResetWindow per Interval stream  (per-stream flow indices)
 + Schedule::with_obs(times)      (the obs boundaries; StepPolicy snap|exact)
      |  log_likelihood(state, obs_idx, params)      (the one scoring seam, gh#139)
      v
{ particle_filter, if2, pmmh, pgas }              (each generic over ObservationModel)
```

Errors flow _alongside_ the data, never as control flow: `bind` always returns
the pair, and the caller decides what `report.verdict` means. `fit`/`pfilter`
refuse on `Error` (a `SimError::Validation` carrying the rendered findings)
unless `--allow-drop[=kind]` downgrades acknowledged kinds; a new
**`camdl check-data <model>
--data …`** runs `bind` purely to render the report
and set an exit code. The private `BoundObs` constructor is the invariant that
makes the whole flow safe.

## Missing data: a hole is marginalized, not imputed

"Missing data" is overloaded, and getting the statistics right matters more here
than anywhere else in the loader — a mishandled missing value is a wrong
likelihood, hence a wrong posterior. There are three genuinely different
situations, and they want three different answers. The first is the common case
and belongs squarely in this proposal; the third is a separate proposal; naming
all three is how we avoid quietly doing the wrong thing for one while solving
another.

### 1. Structurally absent (a hole) — the common case, correct by construction

The overwhelmingly common situation in routine surveillance is that a stream
simply _has no measurement_ at a given time: the weekly cases were reported but
the monthly serosurvey was not, so the serosurvey cell on that week's row is
empty. This is a **hole** — `cells[k] = None` — and the correct thing is what
the bind already does: that stream contributes **no factor** to the joint
likelihood at that time.

Why this is _correct_ and not a shortcut is worth stating plainly, because the
intuition "it's missing, so I should fill it in or sample it" is both common and
costly. In a state-space model the particle filter already carries the full
predictive distribution over the latent state at every time — that _is_ the
particle cloud. Conditioning on an observation reweights the cloud; _not_
conditioning leaves it as the prior-predictive, which is exactly the marginal
over the unobserved value. So **omitting the likelihood factor _is_ the
marginalization**: a proper observation density integrates to one, ∫ p(yₖ | xₖ)
dyₖ = 1, and a factor of one is no factor at all. You do not sample the missing
observation, and there is nothing to sample — the marginalization is free and
automatic. What you must _not_ do is score the hole as an observed zero: a
`None` cell and an observed `0` are different data and must produce different
likelihoods (this is the load-bearing `hole ≠ zero` test). The latent state
still evolves through the hole, driven by the dynamics and by whatever _other_
streams _did_ observe at that time. This is valid under missing-at-random (MAR):
the mere fact that a measurement is absent carries no information about its
value beyond what the model already implies.

So to your question — "do we sample over it?" — for the common case, no, and we
do not need to: the filter marginalizes it by construction, and the
`Option`-cell representation is precisely what lets us write "no factor here"
instead of being forced to invent a value. This is the data layer's whole reason
for existing, and it is **in scope** for this proposal.

### 2. Imputation for reporting (posterior-predictive) — small, additive, opt-in

Sometimes you _do_ want a value where the data is silent — not to condition on,
but to _report_: "given the fit, what would the missing serosurvey plausibly
have read?" That is posterior-predictive imputation: a draw
`ỹ ~ p(y | x_particle, θ)` from the observation model at the hole, on the
**output** side. It changes nothing about the likelihood (it is strictly
downstream of scoring), it reuses the synthetic-data machinery (the reduction
axis below), and it is opt-in. We name it only to keep it clearly separate from
case 1 — imputation is a reporting convenience, never part of the fit — and it
can be added as a flag on the simulate/PPC path without touching `bind`.

### 3. Informative or censored missingness — explicitly out of scope, and why

The hard case is when absence is _not_ random, or when a value is only partially
known: a count censored at a detection limit (`"<5"`, exact value unknown), an
interval-valued report ("between 100 and 200"), below-threshold suppression for
privacy, or — the genuinely informative case — reporting that _fails because
counts are high_ (the surveillance system is overwhelmed at a peak), so a hole
is itself evidence _for_ a large latent value. None of these is MAR, and none is
handled by dropping the factor. A censored observation contributes ∫_region p(y
| xₖ) dy — a real likelihood term integrated over the censoring region, not one
— and an informative-missingness model needs an explicit mechanism p(observed |
xₖ) multiplied in.

This proposal **does not** handle these, and that is a deliberate scope line,
not an oversight. Censoring/truncation is a _new likelihood kind_ (the obs model
integrates its pmf/pdf over a region instead of evaluating it at a point); it
belongs with the observation-likelihood surface, not the data loader, and it is
a separate proposal. What this proposal _does_ contribute toward it: a censoring
mark such as `"<5"` becomes a `RawValue::Unparseable("<5")` → `RejectedValue`
finding, so it is surfaced as a located, named thing the user must resolve,
rather than silently dropped or coerced to `5`. The data layer fails _loudly_ at
exactly the inputs the future censoring proposal will teach it to interpret —
the right interim behaviour.

In one line: MAR holes are in, correct, and the reason we built the `Option`
axis; PPC imputation is a small additive output-side feature; censoring /
informative missingness is a separate likelihood-surface proposal, and until it
lands these inputs are made loud rather than silently wrong.

## The closed-loop hook: `observed_history`

Reactive interventions (the push after this one) close a loop _through_ the
observation layer: a path-B trigger fires on an _observed_ quantity with
reporting noise (`observed(weekly_cases) > 50`), not the latent count. The hook
is one line of scope here: the `Observe` stage maintains an
**`observed_history`** buffer — the most recent observed value per stream — and
exposes it to rate/trigger expressions via an `observed(stream)` primitive. In a
_fit_ the data are given, so reading the actual observed history is
deterministic and free; only _forward / EVSI_ simulation draws a fresh
`y ~ p(y | projection)` to feed the trigger. Naming the buffer now (it is a
trivial write at `Stage::Observe`) is what lets reactive path-B land without
re-plumbing the observation layer later. See the topology proposal's closed-loop
section for the augmented-state treatment.

## `Counted`: the per-survey denominator (malaria)

Binomial slide-positivity — "k positive of n examined" — is the rigorous malaria
prevalence datum, and **n varies survey-to-survey**. Today the
Binomial/BetaBinomial denominator is a model expression
(`BinomialLikelihood { n: Expr }`, `ir/src/observation.rs`), so a survey-varying
n can only be smuggled in as a forcing table — splitting one logical observation
across the model and the data file. The `ObsCell::Counted { value, denom }`
payload fixes this: the denominator rides _with_ the datum in `BoundObs`, and
the likelihood reads it per cell. The fixed-`n: Expr` path stays the default
when no `denom` is supplied. This makes irregular, sparse binomial positivity a
first-class target; only subset-of-strata survey coverage (cross-sectional
surveys rarely cover every cell) waits on the `gh#171` stratum-subset binder,
which is model-side and separate.

## Scope: what the sparse-data substrate unlocks, and what it does not

It is worth being precise about what this proposal makes possible _on its own_,
because "sparse, irregular observations" (gh#171) is a goal with several moving
parts, and the data layer is necessary for all of them but sufficient for only
some. Drawing the line explicitly keeps a reviewer from assuming a target is
reachable when it still needs a model-side companion change.

What it unlocks directly:

- **Sparse, irregular _instant_ observations** — e.g. malaria-style prevalence
  surveys at arbitrary, uneven times. An `Instant` stream reads the state at the
  survey instant (`CurrentPop`/`DerivedExpr`, such as `I/(S+I+R)`), with no
  accumulation window and no per-stream reset; each survey is a present cell on
  the union axis and every non-survey time is a hole contributing nothing. This
  is exactly the bind + `Option`-cell + union-axis case, and it works end to end
  once the union axis is in.
- **Sparse, irregular _interval_ observations** — e.g. monthly incidence
  reported alongside weekly cases — once the per-stream `ResetWindow` (the
  topology work) is wired, so a short stream's observation no longer truncates a
  long stream's accumulation window.
- **Survey-varying Binomial denominators** — via `ObsCell::Counted` (above).
- **Holes scored as marginalized, not as zeros** — the correctness core.

What it does _not_ unlock on its own (named so nobody assumes otherwise):

- **Restricting a stream's `projected` to a subset of strata.** A
  cross-sectional survey rarely covers every age × region cell; expressing "this
  survey only sees these strata" is a _model-side_ change — a stratum-subset
  binder in the DSL — not a data-loading change. The data layer can _represent_
  the resulting holes, but the model must be able to _declare_ the partial
  coverage.
- **Time-varying observation effort.** When reporting effort changes over time
  (a campaign, a stockout, a surveillance scale-up), the right model carries an
  _effort covariate_ (a forcing) into the observation model so the expected
  count scales with effort. That is a forcing + obs-model change, distinct from
  the data being sparse.
- **Censored / informative-missing values** — a separate likelihood-surface
  proposal (the missing-data §, case 3).

So the honest framing for gh#171: this is the data-sparsity _substrate_ —
necessary for every sparse-surveillance target, sufficient on its own for
irregular instant/interval data and survey-varying denominators, and explicitly
_not_ the model-side coverage/effort machinery, which is named and deferred.

## Forward: summary statistics and synthetic likelihood (the reduction axis)

camdl scores every fit through one seam — `ObservationModel::log_likelihood`,
evaluated _per observation time_ and combined sequentially by the particle
filter. That per-cell, Markovian shape is what makes the bootstrap filter and
PGAS work. A whole class of methods deliberately abandons it: **synthetic
likelihood** (Wood 2010; King, Nguyen & Ionides 2016, the pomp `probe_match`
surface) and **ABC**, which score how well _summaries_ of simulated data (peak
height, time-to-peak, final size, growth rate) match summaries of the observed
data. These are the right tool when the per-observation likelihood is
intractable or ill-defined, and they are `gh#172`.

The structural fact: a summary statistic `s(y₁..y_T)` is a function of the
_whole series_, not one `obs_idx`, so it cannot be evaluated inside the
sequential filter. It is **not** a new `ObservationModel` arm — it is a
_sibling_ scorer consumed by a different driver (simulate-many-then-compare, not
sequential weighting). This is the "reduction axis" the topology proposal
reserves:

```rust
trait SeriesScorer {                                  // whole-series, not Markovian
    fn score(&self, observed: &BoundObs, simulated: &[Trajectory], params: &[f64]) -> f64;
}
enum Objective {
    Likelihood(MultiStreamObsModel),   // sequential; consumed INSIDE the PF (today)
    Synthetic(SyntheticLikelihood),    // simulate M reps → N(s; μ_θ, Σ_θ)  (Wood 2010)
    Abc(AbcDistance),                  // accept iff ρ(s_sim, s_obs) ≤ ε
}
```

`BoundObs` gives these methods their input for free: the observed summary is
computed once from `BoundObs`; the simulated summary is computed from each
`Trajectory` projected through the _same_ `StreamProjection`, so the two sides
are apples-to-apples by construction, and holes are handled by the summary
itself ("mean over present cells"). The honest constraint: a summary objective
is **incompatible with PGAS/NUTS** (no per-time conditional density, no latent
path, no gradient) — it composes with the gradient-free outer loops (MH-over-θ,
derivative-free optimization). This proposal does not implement summary scoring;
it only keeps the inference entry points reaching their objective behind that
small abstraction, so adding `Synthetic`/`Abc` is a new _constructor of the
objective_, not a new data path.

## Migration — layered on the topology stages

The data layer is independent of the timeline; only step 3 waits on the topology
work's per-stream `ResetWindow`.

1. **(light, parallel with topology)** `LongRow` parse (long + wide sugar) over
   `caltime`; the NaN/finiteness guard at `pfilter.rs:669/699`. No behaviour
   change.
2. **(light)** `bind` + `BindReport` + `BoundObs`, reproducing today's
   homogeneous/dense semantics so goldens do not move; the report is additive.
   Route the five scattered load sites through it (the unification).
3. **(HEAVY — the correctness tier; gated on the topology `ResetWindow`)** relax
   the ≥5 shared-grid assertions to the union axis + `Option`-cell scoring at
   the single seam `log_likelihood_from_flows_and_counts`
   (`multi_stream_obs.rs`), **wired to** the per-stream `ResetWindow` for
   `Interval` streams. FD/likelihood parity must hold on the dense case; the
   sparse-interval reset gets its own window-correctness test.
4. **(small)** `ObsCell::Counted { value, denom }` through `bind` into the
   Binomial/BetaBinomial scoring path. Scipy-anchored value test; the
   fixed-`n: Expr` path unchanged when no `denom` supplied.
5. **(small)** the `observed_history` buffer + `observed(stream)` primitive (the
   reactive hook), and `check-data` + load-time report + `--allow-drop`.
6. **(cross-cutting)** the `gh#98` calendar equivalence test:
   `expander.parse_date_to_float == caltime::date_to_internal` over a date
   battery (tables convert dates in OCaml at compile, obs-data in Rust at load —
   one constant, two implementations; pin them per the `rata_die` rule).
7. **(separate, model-side)** the `gh#171` stratum-subset binder + effort
   covariate.

## Test obligations (the load-bearing ones)

Every malformed input has a named, located outcome. The non-negotiable
correctness tests:

- **hole ≠ zero**: a `None` cell and an absent row score _identically_; a `None`
  cell and an observed `0` score _differently_. The whole point of the `Option`
  axis — a dedicated likelihood test.
- **sparse `Interval` per-stream reset**: a sparse incidence stream's flow
  accumulated over `[t₁, t₃]` is not truncated by another stream's observation
  at `t₂`. The single most important correctness test, and it exercises the
  topology work's `ResetWindow`.
- **dense-parity regression**: the homogeneous dense case scores bit-identically
  before/after the union-axis refactor (goldens do not move).
- **NaN/inf value cell** → rejected before the log-pmf; **non-numeric value** →
  located `RejectedValue`.
- **invalid-but-date-shaped time cell** (`2024-13-45`, `2024-O3-01`) → located
  `UnparseableDate`, not a silent coercion. This pins the `detect_kind`
  neither-branch that has no test today (the named gap above).
- **censoring mark is surfaced, not coerced**: a `"<5"` value cell becomes a
  located `RejectedValue` finding (not parsed as `5`, not dropped) — the
  loud-interim behaviour until the separate censoring-likelihood proposal lands.
- **`Counted` denom**: scipy-anchored per-cell Binomial/BetaBinomial value test.
- **`Instant` off-grid** (annual prevalence under a daily grid) → snap + warn,
  **not** reject; **`Interval` off-grid** (window can't tile) → error. (The
  snap/exact decision itself is the topology work's `StepPolicy`; this tests the
  _policy choice per kind_.)
- **gh#98** calendar equivalence over a date battery.

## Open questions

- `OffGridInterval` default Error vs a sanctioned, logged `--snap-observations`
  (threading the topology `StepPolicy::Snap` per-stream).
- `--aggregate=sum|mean` for the many:1 case — ship now, or leave to user
  pre-aggregation?
- Where `1:many` aggregate cells come from — does this wait on the deferred
  spatial-aggregation operator?
