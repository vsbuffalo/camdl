---
date: 2026-06-05
status: superseded
superseded_by: 2026-06-06-observation-system.md
related: gh#171, gh#172, gh#98
---

# Observation-data binding

> **SUPERSEDED (archived 2026-06-09)** by `2026-06-06-observation-system.md`,
> which carries this draft's bind/`BoundObs`/typed-holes design forward (and
> folds in three adversarial reviews). Start from
> `2026-06-09-time-and-observation-overview.md`. Retained for history only — do
> not implement from this.

## Framing: bind, not join

A **join** is _symmetric_: two co-equal relations, `A ⋈ B = B ⋈ A`, the result
schema is the union of both attribute sets, and the operation _discovers_ which
keys coincide — neither side is privileged. Loading observation data is not
that. A **bind** is _asymmetric and directional_: the model **defines** a fixed
lattice of named cells (which streams exist; which strata; which grid times
`t_start + k·dt`), and data values are bound _into_ those named slots — like
binding arguments to parameters, or filling a template's holes. You still match
keys mechanically, but the framing privileges the model's lattice as the
authority, and that changes the semantics in exactly the way we need:

- a **leftover** (a data row with no cell — typo'd stream, stratum the model
  lacks, off-grid time) is "data I have nowhere to put" → usually a mistake;
- a **hole** (a cell with no data) is "this slot got nothing" → often _expected_
  (sparse surveillance).

A symmetric join collapses both into one "unmatched" bucket; a bind keeps the
two directions distinct, with distinct severities. And the _result type_
differs: a join yields a union of two tables; a bind yields a **model-shaped
object with typed holes** (`Option` cells) — precisely the sparse-geometry
representation gh#171 needs.

## Why this is a correctness surface

A data point that fails to reach scoring — silently — is a wrong likelihood,
hence a wrong posterior. Genuine residual gaps today (verified against HEAD):

- **NaN/inf are unguarded on the obs path.** `pfilter.rs:669,699` parse values
  with bare `parse::<f64>()`, which accepts `"NaN"`/`"inf"`; nothing checks
  `is_finite` before the value reaches the log-pmf. (Note: gh#100 is a
  _different_ loader — `batch.rs` parameter draws — not this path.)
- **Silent overwrite in the substep map.** `build_obs_at_substep`
  (`pgas.rs:269`) fills a `HashMap<substep, obs_idx>` with `insert` and no
  collision check — two obs rounding to one substep silently drop one from the
  PGAS likelihood. (The CLI load path _does_ hard-error this collision first,
  `caltime_load.rs:254-263`; this is a defense-in-depth latent bug, worth a
  pinning test.)
- **Heterogeneous schedules are a hard wall**, replicated across
  `multi_stream_obs.rs:312-318`, `pfilter.rs:163-179`, `profile.rs:516`,
  `survey.rs:755,890`. Sparse/multi-cadence surveillance (gh#171) can't be
  expressed; the homogeneity is enforced by rejecting, not by reporting.

(Not residual: the `--data` dt-collision is already a clean error
`caltime_load.rs:254-263`; gh#108 is the unrelated `--dates` _output_ render at
`caltime.rs:178`.)

## Unification: one module, consumed as a type

The observation surface is half-unified today. **Scoring** is a single seam —
`MultiStreamObsModel` (`multi_stream_obs.rs`), consumed by PGAS and its
gradient, and since gh#139 inherited by all four methods (PF/IF2/PMMH/PGAS). But
**loading and construction** is scattered and duplicated: the
`--data → per-stream series → MultiStreamObsModel` pipeline is re-implemented in
`pfilter.rs`, `profile.rs`, `fit/runner.rs`, and `survey.rs` — each resolves
`--data` itself, builds its own `per_stream_obs`, runs the shared-grid check
(replicated across ≥5 sites), and canonicalizes
`observations = per_stream_obs[0]`. That duplication is where the silent drops
and the homogeneity asserts live.

So `obsdata::bind` is not only an audit — it is the **unification**: it owns the
load + validate + construct, emits one `BoundObs`, and every algorithm _consumes
that type_ instead of re-deriving it. One module, one type, defined once — so
the on-grid policy, the temporal-kind handling, and the accumulator-reset
semantics live in a single place, not per caller. The data _flow_ becomes
`load → bind → BoundObs → {PF, IF2, PMMH, PGAS}`, rather than each algorithm
re-loading and re-checking.

## Temporal kind is first-class

Each stream has a `TemporalKind`, derived from its projection, and it governs
on-grid policy, accumulator semantics, and off-grid handling:

- **`Interval`** (`CumulativeFlow`/`CumulativeFlowSum`, i.e. incidence): the
  value is flow accumulated over `[prev_obs, this_obs]`; the cell is a window.
  - On-grid is a **correctness** requirement: `dt` must tile the window, which
    holds iff both endpoints are on the grid → off-grid is an **error**.
  - Needs **per-stream accumulator reset** at that stream's obs times. The
    current reset is _global_ (`particle_filter.rs:401`), which over a sparse
    stream would inflate other streams' windows — this is the umbrella's §5.2.1
    CRITICAL finding and must land _with_ `None`-skip, not after.
- **`Instant`** (`CurrentPop`/`CurrentPopSum`/`DerivedExpr`, i.e. prevalence):
  the value is an instantaneous snapshot read at the obs time
  (`resets_after_observation` is false, `multi_stream_obs.rs:84-89`). There is
  no window to tile.
  - Off-grid is a mild **snapshot-time** error → read at the nearest grid point
    and **warn** past a tolerance, do _not_ reject.
  - No accumulator, no reset.

A single unified "off-grid = error" rule (the prior draft's mistake) would
hard-reject a valid annual prevalence survey under a daily grid. The kind split
fixes that.

## The bind, as a cardinality map

`bind` is a partial map `φ : DataRow → ModelCell`, cell =
`(stream, stratum, k)`. Every failure is a departure from "injective and total,"
with a defined resolution:

| cardinality            | cause                                        | resolution                                                                                     |
| ---------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 1:1                    | —                                            | clean                                                                                          |
| many:1 (non-injective) | `dt` coarser than data, or duplicate row     | `Collision`/`Duplicate` → **Error**; `--aggregate=sum\|mean` opt-in (loud, changes likelihood) |
| 1:many                 | data coarser than model (region vs district) | needs a model aggregate cell (`CumulativeFlowSum`); else `CoarserThanModel` → **Error**        |
| 0:1                    | cell with no data — a hole                   | `None` cell; `Hole` → **Info** (sparse) / **Warn** (stream declared dense)                     |
| 1:0                    | data with no cell — a leftover               | `LeftoverColumn`/`LeftoverStratum` → **Error**; benign extra metadata column → **Info**        |

Severities split by **direction**: data-has-extra (`LeftoverColumn`) defaults to
Info (real files carry `population`, `notes` columns); model-cell-unfilled-
when-dense and stratum-mismatch default to Error.

## Types and how they flow into one another

Two type families meet at the bind: the **new loader types** (`obsdata`, below)
and the **existing inference types** they feed. The value of the bind is that it
is the _one_ place those families connect — data enters as untyped rows and
leaves as a value the inference traits already know how to consume. Nothing
downstream is new; the work is to give the existing seam a single, typed input.

### The new types (`obsdata`)

```rust
mod obsdata {
    /// Set by the stream's projection; governs grid policy + accumulator
    /// semantics. The runtime analogue is `StreamProjection` (below).
    pub enum TemporalKind { Interval, Instant }

    // ── input: one untyped row per PRESENT observation ──
    struct LongRow { stream: String, stratum: Option<String>, when: RawTime, value: RawValue }
    enum RawTime  { Offset(f64), Date(String) }      // resolved via ir::caltime + model origin
    enum RawValue { Num(f64), Missing, Unparseable(String) }

    // ── output: a model-shaped, fully-typed object ──
    /// PRIVATE ctor — only `bind` constructs one, so no un-validated data can
    /// reach the likelihood.
    pub struct BoundObs { times: Vec<f64>, streams: Vec<StreamCells> }
    struct StreamCells {
        name:  String,
        kind:  TemporalKind,
        cells: Vec<Option<ObsCell>>,    // None = hole; one slot per time in `times`
    }
    /// Payload of a present cell. `Scalar` is the common case; `Counted`
    /// carries per-observation auxiliary data — a Binomial/BetaBinomial
    /// denominator that varies survey-to-survey (the malaria case, §Scope).
    /// Both ship in this proposal: the cell is `Option<ObsCell>`, so holes stay
    /// typed regardless of payload, and adding a future payload variant stays
    /// additive.
    enum ObsCell { Scalar(f64), Counted { value: f64, denom: f64 } }

    // ── the report: errors are VALUES, not control flow ──
    pub enum Severity { Error, Warn, Info }
    pub struct Finding  { kind: BindIssue, stream: String, detail: String, count: usize, severity: Severity }
    pub enum BindIssue {
        LeftoverColumn, LeftoverStratum, OffGridInterval, OffGridInstant,
        Collision, Duplicate, CoarserThanModel, Hole, RejectedValue,
        UnparseableDate, InconsistentTimeColumn,   // surfaced from the time-column typer
    }
    pub struct BindReport { findings: Vec<Finding>, verdict: Severity }

    pub fn bind(model: &Model, rows: Vec<LongRow>, dt: f64, cal: &CalendarCtx, policy: &BindPolicy)
        -> (BoundObs, BindReport);   // never panics, never exits — errors are VALUES (gh#181)
}
```

### The existing types it feeds (verified against the code)

`BoundObs` is not a parallel universe — it is the input to types that already
exist in `sim::inference`. The seam is small and already unified for _scoring_;
this proposal unifies what _builds_ it.

- **`trait ObservationModel<S>`** (`traits.rs:89`) — the single seam every
  algorithm scores through. Its core method is
  `fn log_likelihood(&self, state: &S, obs_idx: usize, params: &[f64]) -> f64`
  (`traits.rs:94`), plus `n_observations` / `obs_time(obs_idx)` / `n_streams`.
  The doc-comment is explicit: "This is the ONLY method required for inference.
  All algorithms (PF, IF2, PMMH, PGAS) call this for particle weighting." This
  is the type every fitting algorithm consumes — exactly the "passed as a type
  to each of them" shape.
- **`struct MultiStreamObsModel`** (`multi_stream_obs.rs:246`) — the production
  `impl ObservationModel<ParticleState>`. Owns `obs_times`, the per-stream
  `StreamProjection`, and the per-stream observed series. This is what `bind`'s
  output is _built into_.
- **`enum StreamProjection`** (`multi_stream_obs.rs:72`) —
  `FlowSum | IntCompSum | Expr`, with `resets_after_observation()` true only for
  `FlowSum`. This **is** `TemporalKind` at the runtime layer:
  `Interval ≙ FlowSum` (reads `flow_accumulators`, resets);
  `Instant ≙ IntCompSum`/`Expr` (reads `counts`, no reset). `bind` chooses the
  variant; the projection enforces the semantics.
- **`struct ParticleState { counts, flow_accumulators }`** (`types.rs`) +
  **`trait Resettable`** (`traits.rs:27`) — the state a projection reads.
  `Resettable::reset_accumulators` is the per-stream-reset hook the
  sparse-`Interval` work needs; today it is driven _globally_
  (`particle_filter.rs:401-402` resets every particle's accumulators at every
  obs time), which is the umbrella's §5.2.1 trap for sparse incidence.
- **`trait ProcessModel` / `DensityProcess`** (`traits.rs`) — the simulator
  side; unchanged here, named only to locate the observation seam relative to it
  (`ProcessModel::State: Resettable` is the bound that ties the two).

So `BoundObs` slots in at exactly one place, and the mapping into
`MultiStreamObsModel` is mechanical:

| `BoundObs`               | becomes, in `MultiStreamObsModel`                                                                                    |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `times` (the union axis) | `obs_times`; drives `n_observations()`                                                                               |
| `StreamCells.kind`       | the `StreamProjection` variant (Interval→`FlowSum`; Instant→`IntCompSum`/`Expr`)                                     |
| `cells[k] = Some(v)`     | a scored observation for that stream at `obs_idx = k`                                                                |
| `cells[k] = None` (hole) | that stream contributes **0** to the joint log-likelihood at `obs_idx = k` — skipped, not scored as an observed zero |

That last row is the entire correctness point: a hole is the _absence of a term
in the sum_, not an observed value of zero. The homogeneous path can't express
it because every stream shares one dense `obs_times`; `Option` cells over a
union axis can.

### The flow

```
--data PATH  /  --data NAME=PATH                  (raw file: long, or wide-sugar)
      |   parse + ir::caltime  (date -> model-time)
      v
Vec<LongRow>                                      (untyped: stream, stratum, when, value)
      |   obsdata::bind(model, rows, dt, cal, policy)
      v
(BoundObs, BindReport) ----------------> report.verdict
      |  value: model-shaped, typed,             |  Error      -> refuse: SimError::Validation
      |  Option cells (holes typed)              |               with rendered findings,
      |                                          |               unless --allow-drop[=kind]
      |                                          |  Warn/Info  -> proceed, surface findings
      v
MultiStreamObsModel : ObservationModel<ParticleState>
      |   log_likelihood(state, obs_idx, params)   (the one seam; traits.rs:94)
      v
{ particle_filter, if2, pmmh, pgas }              (each generic over ObservationModel)
      |   reads state via StreamProjection
      v
ParticleState { counts (Instant) | flow_accumulators (Interval) }
      ^   Resettable::reset_accumulators          (per-stream, for Interval streams)
```

Errors flow _alongside_ the data, never as control flow. `bind` always returns
the pair; the caller decides what `report.verdict` means: `fit`/`pfilter` at
load refuse on `Error` (a `SimError::Validation` value carrying the rendered
findings) unless `--allow-drop[=kind]` downgrades acknowledged kinds, and the
new **`camdl check-data <model> --data …`** subcommand runs `bind` purely to
render the report and set an exit code. (`check-data` is a _new Rust_
subcommand, **not** the OCaml `check`, which is a `camdlc` passthrough at
`main.rs:206,414` and never reads obs data.) Findings render as structured
diagnostics so `--json-errors` / CI / the book consume them (gh#181). The
invariant that makes the whole flow safe: `BoundObs` has a private constructor,
so the only way to obtain one is through `bind`, so every leftover, collision,
and hole is accounted for in some `BindReport` before any value reaches
`log_likelihood`.

## Input format and column typing

One row per _present_ observation: `time, stream, stratum, value`. Sparsity is
absent rows — no NaN-vs-zero ambiguity. A long-indices/wide-streams sugar
(`time, patch, afp, es`) is accepted and normalized in, with empty = typed
`None`, never zero. `BoundObs` is itself "long indices × wide `Option` cells".

The robustness question — _how does the binder know which column is which, and
how robust is date handling?_ — has two layers, and one of them is already
solid.

### Column **role** is bound by name, never sniffed from content

A column's role (time / stream / stratum / value) is decided by **matching the
model's declared names**, not by guessing from cell contents. The model is the
schema (the "bind, not join" principle applied to columns):

- the **time** column is identified by a fixed header (`time`) or `--time-col`,
  not by "which column looks date-shaped";
- a header matching a model **stream** name _is_ that stream; a header matching
  a **stratum** dimension _is_ that stratum (the multi-stream "column-named
  loader" already works this way — every column must be named);
- a header matching **nothing** is a `LeftoverColumn` finding — located,
  surfaced, Info if it looks like benign metadata (`population`, `notes`), never
  silently dropped and never content-sniffed into a role.

So "what if it can't figure out a column's type?" has a definite answer: it does
not guess. An unrecognized column is reported with the column name and a hint
listing the model's known stream/stratum names. Mis-binding by content heuristic
— the failure mode that silently routes the wrong column into the likelihood —
cannot happen.

### Time **cell** typing is already auto + robust (reused, not reinvented)

The whole-column time typer already exists and is the thing `bind` calls — it is
not new surface. `caltime_load::convert_time_column` /`detect_kind`
(`caltime_load.rs:100-221`):

- scans the **whole column**: all cells numeric → numeric (day-offsets); all
  cells ISO-date → dated (converted via the model's `origin` + `time_unit`);
- a **mixed** column → hard error naming _both_ offending rows
  (`caltime_load.rs:131-140`; tested: `mixed_column_errors_naming_both_rows`);
- a **`--time-format numeric|date`** override is honoured _before_ detection
  (`:154-158`), with a helpful message when `numeric` meets a date cell;
- a dated column with **no model `origin`** → hard error with the fix hint
  (`:206-211`; tested: `dated_without_origin_errors`).

So "a date-like column with one bad date in the middle" is already a _located
hard error_, not a silent coercion: if the bad cell parses as a number the typer
reports a mixed column naming both rows; if it parses as neither a number nor an
ISO date, the `detect_kind` else-branch (`:112-125`) reports
`line N: time cell '…' is neither a number nor an ISO date (reason)`. Under the
bind, these become `Finding`s (`UnparseableDate` / `InconsistentTimeColumn`,
Error severity) rather than bare `Result::Err`, so they flow through
`BindReport` with everything else — but the detection and the located message
already exist.

The honest gap is the **invalid-but-date-shaped** cell (`2024-13-45`,
`2024-O3-01` with a letter O): it hits the right else-branch but there is **no
test pinning it** today (the suite covers numeric+date mix, not the
neither-branch). That is a test obligation below, not a missing mechanism.

### Value **cell** typing is the part that needs hardening

`caltime_load` is time-only. The observation **value** cells are parsed with a
bare `parse::<f64>()` (`pfilter.rs:669,699`) that accepts `"NaN"`/`"inf"` and
has no located error for a non-numeric value. `bind` adds the typed `RawValue`
(`Num | Missing | Unparseable`) and a finiteness guard, so a bad value cell is a
located `RejectedValue` finding and a `NaN`/`inf` can never reach the log-pmf.
This — not the date path — is where robustness is actually added.

## Scope: what this unlocks (and what it doesn't)

This is the **data-sparsity substrate** for gh#171: it lets the _data_ be sparse
and time-varying, and the `Option` cells carry it through scoring. It does
**not** by itself satisfy gh#171's two model-side asks — restricting a stream's
`projected` to a subset of strata, and time-varying observation effort (a
forcing). Those are separate (a subset binder in the DSL; an effort covariate).
Necessary, not sufficient — the Sokoto ES case needs all three. gh#172
(summary-statistic targets) is orthogonal (it changes _what_ is scored; see
§Forward).

### Worked target: malaria-like sparse, irregular prevalence

Does this unlock fitting to malaria-style data — prevalence surveys at sparse,
irregular times? Mostly **yes**, and the part it doesn't reach is named.

- **The hard part — yes.** Malaria prevalence is an `Instant` stream
  (`CurrentPop`/`DerivedExpr`, e.g. `I/(S+I+R)`): read at the survey instant, no
  accumulation window, no per-stream reset (it bypasses `flow_accumulators`
  entirely and reads `counts`). Sparse + irregular survey times are exactly the
  union-axis + `Option`-cell case: each survey is a present cell on the union
  `times`; every non-survey time is a typed hole that contributes no term.
  Off-grid survey dates resolve to the nearest grid point and **warn** (not
  reject), per the `Instant` policy. So the binding and scoring of irregular
  sparse prevalence is precisely what this proposal delivers.
- **Continuous prevalence index — fully covered.** If the datum is a proportion
  scored with Normal/Beta on `I/N`, one scalar per cell suffices; nothing else
  is needed.
- **Binomial slide-positivity — the `Counted` cell, in scope here.** The
  rigorous datum is "k positive of n examined," and **n varies
  survey-to-survey**. Today the Binomial/BetaBinomial denominator is a model
  expression (`BinomialLikelihood { n: Expr }`, `crates/ir/src/observation.rs`),
  so a survey-varying n can only be smuggled in as a forcing table indexed by
  survey time — which splits one logical observation (k of n at t) across the
  model and the data file. The fix is the `ObsCell::Counted { value, denom }`
  payload: the denominator rides _with_ the datum in `BoundObs`, and the
  likelihood reads it per cell. This ships in this proposal (it is a small,
  self-contained addition to the cell type and the Binomial/BetaBinomial scoring
  path), so binomial positivity is a first-class target, not a deferral.
- **The model-side gaps (same as #171).** If prevalence is observed only in a
  subset of patches/age-strata (cross-sectional surveys rarely cover every
  cell), that restriction is the gh#171 subset binder — model-side, separate
  from this loader.

Net: a malaria fit with continuous prevalence _or_ binomial slide-positivity at
irregular sparse times works on this proposal (continuous via `Scalar`,
positivity via `Counted`); only subset-of-strata coverage waits on gh#171.

## The shared calendar (gh#98)

`bind` converts dates via `ir::caltime` (the model's time basis). But tables
convert dates in **OCaml** (`expander.ml:130` `parse_date_to_float`) at compile
time, while obs-data converts in **Rust** (`caltime`) at load — two
implementations of one constant. Pin them with an equivalence test
(`expander.parse_date_to_float == caltime::date_to_internal` over a date
battery), per the `rata_die` cross-language rule.

## Migration (honest about the heavy tier)

1. **(light)** `LongRow` parse (long + wide sugar) over `caltime`; the NaN guard
   at `pfilter.rs:669/699`; the `build_obs_at_substep` collision pin. No
   behavior change.
2. **(light)** `bind` + `BindReport` + `BoundObs`, reproducing today's
   homogeneous/dense semantics so goldens don't move; the report is additive.
3. **(HEAVY — the real correctness tier, not "incremental")** Relax the ≥5
   shared-grid assertions to the union axis + `Option`-cell scoring at the
   single likelihood seam `log_likelihood_from_flows_and_counts`
   (`multi_stream_obs.rs:366-391`, the ~100×-divergence code), **together with**
   per-stream accumulator reset for `Interval` streams. FD/likelihood parity
   tests must hold on the dense case; the sparse-interval reset needs its own
   window-correctness test (the umbrella's §5.2.1 trap).
4. **(small)** `ObsCell::Counted { value, denom }` — carry a per-observation
   Binomial/BetaBinomial denominator through `bind` into the scoring path; the
   likelihood reads `denom` per cell instead of evaluating the model `n: Expr`.
   Scipy-anchored value test on the per-cell-`n` path; the existing
   fixed-`n: Expr` path stays the default when no `denom` is supplied.
5. `check-data` + load-time report + `--allow-drop`.
6. The gh#98 date-equivalence test.
7. (separate) the gh#171 subset binder + effort covariate — model-side, not this
   proposal.

## Forward: summary statistics and synthetic likelihoods

camdl scores every fit through one seam — `ObservationModel::log_likelihood`,
evaluated _per observation time_ and combined sequentially by the particle
filter. That per-cell, Markovian shape is what makes the bootstrap filter and
PGAS work. But a whole class of methods deliberately abandons the per-time
likelihood: **probe-matching / synthetic likelihood** (Wood 2010; King, Nguyen &
Ionides 2016, the pomp `probe`/`probe_match` and synthetic-likelihood surface)
and **approximate Bayesian computation (ABC)**. These score a model by how well
_summaries_ of simulated data match summaries of the observed data — peak
height, time-to-peak, final size, growth rate, autocorrelations, spectral
features — rather than by a point-by-point density. They are the right tool when
the per-observation likelihood is intractable, ill-defined, or pathological
(near-deterministic dynamics, hard-to-specify reporting processes), and they are
on the roadmap as gh#172. This proposal should not implement them, but it should
not foreclose them — and a few choices here decide whether they slot in cleanly.

**The structural fact: a summary statistic is a function of the whole series,
not of one `obs_idx`.** `s(y₁..y_T)` cannot be evaluated inside the sequential
filter; it needs the full trajectory in hand. So summary-stat scoring is **not**
a new `ObservationModel` arm — it is a _sibling_ scorer consumed by a different
driver (simulate-many-then-compare, not sequential weighting):

```rust
/// Today's per-cell sequential likelihood (MultiStreamObsModel) and a future
/// whole-series summary scorer are SIBLINGS, not the same trait. Both read the
/// same `BoundObs` — the data type does not fork; the scorer does.
trait SeriesScorer {
    /// A (pseudo-)log-density of the observed data given simulated output.
    /// Sees the WHOLE series, not one obs_idx — not sequential, not Markovian.
    fn score(&self, observed: &BoundObs, simulated: &[Trajectory], params: &[f64]) -> f64;
}

enum Objective {
    Likelihood(MultiStreamObsModel),   // sequential; consumed INSIDE the PF (today)
    Synthetic(SyntheticLikelihood),    // simulate M reps → N(s; μ_θ, Σ_θ)  (Wood 2010)
    Abc(AbcDistance),                  // accept iff ρ(s_sim, s_obs) ≤ ε
}
```

**What `BoundObs` gives these methods for free.** The observed summary
`s(y_obs)` is computed _once_ from `BoundObs`; the simulated summary is computed
from each simulated `Trajectory` **projected through the same
`StreamProjection`**. Because both sides flow through the identical projection
(the `Interval`/`Instant` split, the stratum layout, the union axis), the
observed and simulated summaries are guaranteed apples-to-apples — a summary
function is just a reduction over the model-shaped cells, defined once and
applied to both sides. Holes (`None` cells) are handled by the summary itself
(e.g. "mean over present cells"), exactly as they should be, with no special
path. Synthetic likelihood then fits a Gaussian to the M simulated summary
vectors and scores `s(y_obs)` under it (Wood 2010); ABC thresholds a distance.
Neither needs a gradient, and neither touches the loader.

**How it plugs into a fit — concretely.** A fit today is an _outer parameter
search_ wrapped around an _inner objective_ that turns a θ into a scalar
(pseudo-)log-density. The inner objective is built once —
`FitConfig::build_obs_model` (`fit/runner.rs:404`) constructs the
`MultiStreamObsModel` — and handed to the algorithm, which scores it
**sequentially inside a particle filter**: `if2` and `particle_filter` take it
as `&dyn ObservationModel<ParticleState>`, `pgas`/`pmmh` take the concrete
`&MultiStreamObsModel` (PGAS additionally needs the per-state gradient and a
latent-state representation). A `SeriesScorer` replaces the _inner objective
only_: instead of "run a PF to get a marginal likelihood," it is "simulate M
trajectories at θ, project each through the same `StreamProjection`, reduce to
summaries, score against the observed summaries." That scalar then feeds the
**same** kind of outer search. So the integration is a new `algorithm` value in
`fit.toml` (e.g. `algorithm = "probe_match"` / `"synthetic"` / `"abc"`) whose
objective is a `SeriesScorer` and whose outer loop is a derivative-free
optimizer (Nelder-Mead, as pomp's `probe_match` uses) or a Metropolis sampler
over θ — reusing the existing `EstimatedParam`/`Transform` bounds-and-scale
machinery unchanged.

The honest constraint: a summary objective is **incompatible with PGAS and
NUTS**. Those need a per-observation-time conditional density, a latent-state
path, and a gradient — none of which a whole-series summary provides. Summary
methods compose with the _gradient-free_ outer loops (MH-over-θ, derivative-free
optimization), not with the sequential/gradient samplers. That is not a
limitation of this proposal; it is intrinsic to summary-based inference, and
naming it is what keeps the `Objective` enum honest about which (driver,
objective) pairs are valid.

**What to keep clean now** is only that inference entry points reach their
objective behind that small abstraction rather than hard-wiring
`MultiStreamObsModel`. The §Unification refactor — routing every algorithm
through one constructed obs type built from `BoundObs` — is exactly what makes
adding `Synthetic`/`Abc` a new constructor of the objective rather than a new
data path. The data binding is the foundation; the scorer fork sits one layer
above it, in the fit driver.

## Robustness & test obligations (the checklist)

The point of a typed bind is that every malformed input has a _named, located_
outcome. Each row below is a test; `[have]` already exists, `[gap]` is new work
this proposal owes. No silent coercion, no silent drop, anywhere.

**Column typing / parsing**

- `[have]` mixed numeric+date time column → error naming both rows
  (`mixed_column_errors_naming_both_rows`).
- `[have]` dated column with no model `origin` → error with fix hint
  (`dated_without_origin_errors`).
- `[have]` `--time-format numeric` forbids date cells; `date` forces conversion.
- `[gap]` **invalid-but-date-shaped time cell** (`2024-13-45`, `2024-O3-01`) →
  located `UnparseableDate` error. The else-branch handles it
  (`caltime_load.rs:112-125`) but nothing pins it.
- `[gap]` non-numeric **value** cell → located `RejectedValue` finding (not a
  panic, not a silent skip).
- `[gap]` `"NaN"`/`"inf"` value cell → rejected before the log-pmf (the
  `pfilter.rs:669,699` finiteness guard).
- `[gap]` header matching no model stream/stratum/time → `LeftoverColumn`,
  surfaced (Info), never silently dropped.
- `[gap]` wide-sugar empty cell → typed `None` hole, never `0`.

**Cardinality / bind correctness**

- `[gap]` two rows → one cell (many:1) → `Collision` Error — the
  `build_obs_at_substep` silent-overwrite pin (`pgas.rs:269`).
- `[gap]` duplicate identical row → `Duplicate`.
- `[gap]` **hole ≠ zero**: a `None` cell and an absent row score _identically_;
  a `None` cell and an observed `0` score _differently_. This is the whole
  correctness claim of the `Option` axis — it gets a dedicated likelihood test.
- `[gap]` leftover stream/stratum (1:0) → `LeftoverStratum` Error.
- `[gap]` data coarser than model (1:many) → `CoarserThanModel` Error.

**Temporal kind**

- `[gap]` `Instant` off-grid (annual prevalence under a daily grid) → snap +
  warn, **not** reject.
- `[gap]` `Interval` off-grid (window can't tile) → Error.
- `[gap]` **sparse `Interval` per-stream reset** — the §5.2.1 trap: a sparse
  incidence stream's flow accumulated over `[t₁, t₃]` is not truncated by
  another stream's observation at `t₂`. The single most important correctness
  test here.

**`Counted` payload**

- `[gap]` per-cell Binomial/BetaBinomial denominator: scipy-anchored value test;
  `Counted.denom` used instead of the model `n: Expr`; the fixed-`n: Expr` path
  unchanged when no `denom` is supplied.

**Cross-cutting**

- `[gap]` gh#98 calendar equivalence:
  `expander.parse_date_to_float == caltime::date_to_internal` over a date
  battery.
- `[gap]` dense-parity regression: the homogeneous dense case scores
  bit-identically before/after the union-axis refactor (goldens do not move).

## Open questions

- `OffGridInterval` default Error vs a sanctioned, logged `--snap-observations`.
- `--aggregate=sum|mean` for the many:1 case — ship now or leave to user
  pre-aggregation?
- Where `1:many` aggregate cells come from — does this wait on the deferred
  spatial-aggregation operator (umbrella §7.1)?
