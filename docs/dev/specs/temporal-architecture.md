# Temporal architecture for forecasting and predictive scoring

**Status:** Draft for review (2026-11-15). Nothing here is implemented. The
migration that would implement it, with the audit it rests on, is
`docs/dev/proposals/2026-11-15-temporal-architecture-migration.md` (the
migration proposal). Where a rule here changes current behaviour, the migration
proposal lists the change and the test that pins it.

**Builds on, does not replace:** `docs/dates.md` (the calendar policy),
`docs/dev/proposals/2026-09-05-observation-time-as-a-sum-type.md` (`Period`,
`StreamTimes`, `covers`; accepted and landed),
`docs/dev/proposals/2026-11-14-score-artifact.md` (the score artifact, row
digests, `TimeAxis`, `ScoredLeadTime`; proposal) and
`docs/dev/proposals/2026-08-29-honest-predictive-evaluation.md` (§3.7 tail
holdout, §3.7.2 forecast mode; Stages 1–4 landed). Vocabulary is taken from
those documents wherever they already name a thing.

## 1. The question this answers

A forecast score is a statement of the form: _with parameters fitted to these
observations, and the latent state conditioned on these observations, the model
predicted that this quantity, measured over this period, would have this
distribution; the observation was this._ Each italicised noun is a temporal
object. Today several of them are carried as one `f64` and reconstructed by
convention at each consumer, and the conventions disagree (migration proposal,
§2). This document fixes one meaning for each, names the code layer that owns
it, and gives the smallest set of runtime types that keeps the meanings apart
where swapping them would produce a plausible wrong number.

The answer in one paragraph: **every temporal decision about an observation is
made on its support — the instant or period it measures — and never on its row
label; the training/held-out partition is a per-leaf prefix of rows, decided
once from the support, recorded with its row count and digest, and consumed by
every later command; and every prediction is made against an explicit, validated
list of targets, each a (leaf, support) pair, built once in a `ForecastPlan`.**
The simulation engines keep raw `f64` times.

## 2. Core concepts

Each concept below has one name. Documentation, CLI help, artifact field names
and code identifiers use that name; §9 lists the names that are retired.

### 2.1 Axis concepts (exist today; unchanged)

- **Model time** `t`. A real number in the model's `time_unit`. The only time
  representation inside the engine. Owned by the IR.
- **Calendar axis.** The pair (`time_unit`, `origin`), where `origin` is the
  optional calendar instant mapped to `t = 0` (`docs/dates.md`, "The one rule").
  A model with an `origin` is _anchored_. The calendar axis is a property of the
  model, never of a forecast. The word **origin** unqualified always means this.
- **Instant, duration, period.** As in `docs/dates.md`, "Three concepts". A
  period is half-open `[start, stop)` externally.
- **Exact and calendar durations.** As in `docs/dates.md`. A calendar duration
  (`'months`, `'years`) never translates an instant except through the named
  functions `add_calendar_months` / `add_calendar_years`.

### 2.2 Observation concepts

- **Row.** One line of a bound data file, routed to one leaf (a concrete
  observation stream; `2026-11-13-compare-on-selected-streams.md`). A row has a
  value or is a hole (`NA`).
- **Label.** The number or date written in the row's time column. A label is
  data, not a time the model uses. It survives only as provenance (the row
  digest hashes it as written).
- **Support.** What the row measures: an instant `t` (a state read,
  `projected =
  I`) or a period `[start, stop)` (a flow, `incidence(...)`). For
  a uniform `covers` form it is `Covers::period_of(label)`
  (`ir/src/observation.rs:304`); for `window_start`/`window_stop` columns it is
  read per row. The support is computed in exactly one place and is the only
  temporal attribute of a row that any later decision reads.
- **Close.** The instant at which a row's information is complete: `t` for an
  instant support, `stop` for a period. It is the row's scoring time
  (`StreamTimes::closes`, `multi_stream_obs.rs:1520`). "Scored at" in the
  published page is the close.
- **Open.** `start` for a period, `t` for an instant.
- **Row position.** The 0-based index of a row among its leaf's rows, in file
  order. File order is strictly increasing in close for every bound leaf
  (enforced at `BoundObs::bind`, `multi_stream_obs.rs:1003`), so position is
  also time order. Position is the row identity this design uses (§5).

### 2.3 Conditioning concepts

- **Conditioning cutoff** `τ`. An instant: a row is inside an information set
  bounded by `τ` iff its close ≤ `τ`. Relative to an instant `u`, an untrained
  row is _after_ `u` when its open ≥ `u` and _straddles_ `u` when open < `u` <
  close. These are the only rules by which a time partitions rows (§5.2).
  `holdout_after = τ` declares a cutoff.
- **Parameter information set** `Iθ`. The rows the parameter estimate was fitted
  to. Recorded by the fit as a `TrainingPartition` (§5).
- **State information set** `Ix`. The rows the latent state at the forecast
  origin was conditioned on (assimilated by a filter or smoother). For a
  fixed-origin forecast `Ix ⊇ Iθ` is allowed — a fit sealed last week whose
  state is filtered through this week's data — and the two are recorded
  separately. For a free-forward projection `Ix = ∅`.
- **Training extent.** The latest close over `Iθ`, over every leaf. It is the
  instant up to which the parameters have seen data. It is what
  `training_window.train_end` records (the score-artifact proposal's "latest
  training row", made precise as a close, not a label; §5.3).

### 2.4 Forecast concepts

- **Forecast origin** `s`. The instant from which predictive dynamics run with
  no further assimilation. Never called "origin" alone; always "forecast
  origin". Default `s =` the close of the latest row in `Ix`.
- **Target.** A `(leaf, support)` pair the forecast predicts, with
  `open(support) ≥ s`. A target exists whether or not an observation for it
  exists; when one does, the target carries the row's position.
- **Lead time** of a target: `close(support) − s`, a duration in model units.
  This is the score-artifact proposal's "lead time" measured from a forecast
  origin, which that proposal reserves for gh#808.
- **Forecast horizon** `H`. A duration: the maximum lead time requested. Targets
  are exactly those with `close(support) ≤ s + H`. The horizon bounds the
  forecast; it never sets the cadence of targets (§6.2). This is the one meaning
  of "horizon" (§9).
- **Issue time.** When a forecast was produced (wall clock). Recorded as
  provenance only; it is not the forecast origin. A real-time forecast issued on
  a Monday for data closing the previous Saturday has `s =` Saturday's close.
- **As-of date** of a data file: the author's statement of when its contents
  were current (score-artifact proposal, "The dataset label and the as-of
  date"). Unhashed provenance. This design does not use it to decide membership.

### 2.5 Predictive mode

Three scientifically distinct predictive objects, independent of backend:

| Mode           | Parameters                             | Latent state at a target                                    | What a score means                                  |
| -------------- | -------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| `InSample`     | fitted to rows that include the target | filtered through the previous row                           | goodness of fit; optimistic                         |
| `OneStepAhead` | `Iθ` excludes the target               | filtered through every earlier row, including held-out ones | one-observation-ahead skill on unseen data (§3.7.1) |
| `FixedOrigin`  | `Iθ` excludes the target               | conditioned through `s` only; free-running after            | forecast skill at lead time `close − s` (§3.7.2)    |

A free-forward projection from the initial state (today's
`fit predict
--horizon free_forward`) is `FixedOrigin` with `s = t_start` and
`Ix = ∅`: the same object with an empty state information set, not a fourth
mode.

Parameter uncertainty (a plug-in point or a posterior mixture: the trace's
`Provenance`) is a second, orthogonal axis, as `2026-08-29` §3.2 already types
it. The numerical method (ODE skeleton plus observation noise, chain-binomial
particle filter, Gillespie) is a third axis, recorded as the trace's `basis`
(`score/src/trace.rs:61`). No combination of the three axes is implied by
another.

## 3. Invariants and who owns them

"Static" means a value violating the invariant cannot be constructed; "dynamic"
means a constructor returns an error; "convention" means only a comment says so.
No invariant in this table is left to convention.

| #   | Invariant                                                                                                                                                                              | Owner                                                    | Kind                                                    |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------- |
| I1  | A calendar duration never translates an instant except via `add_calendar_*`, in every DSL position, including `covers` widths                                                          | OCaml `time_typing.ml`                                   | static (compile error E321)                             |
| I2  | `Instant + Instant`, `k · Instant`, `Duration − Instant` are refused; `Instant − Instant` is a Duration                                                                                | OCaml `time_typing.ml`                                   | static (compile error)                                  |
| I3  | `date_range(d, …, calendar_months = n)` yields `add_calendar_months(d, i·n)` for each `i`, computed from `d`                                                                           | OCaml expander                                           | static (by construction)                                |
| I4  | The calendar axis (`time_unit`, `origin`) of a model is carried, unchanged, into every artifact derived from it                                                                        | Rust `ir::caltime::CalendarAxis`                         | dynamic (artifact constructors take it)                 |
| I5  | A period has finite, ordered endpoints                                                                                                                                                 | Rust `Period::new`                                       | dynamic (exists)                                        |
| I6  | A row's support is computed only by `observed_rows` (one reader)                                                                                                                       | Rust `cli::fit::observed_rows`                           | static (the only producer of `ObservedRow`)             |
| I7  | Partitions by time read supports, never labels                                                                                                                                         | Rust `TrainingPartition::from_cutoff`                    | static (takes `&[ObservedRow]`, has no label parameter) |
| I8  | Per leaf, a training partition is a prefix of the leaf's rows                                                                                                                          | Rust `TrainingPartition`                                 | dynamic (constructor)                                   |
| I9  | A held-out score scores only rows after the training extent, and the rows before it digest to the fit's recorded prefix                                                                | Rust `score` preflight                                   | dynamic (refusal)                                       |
| I10 | A target's support opens at or after the forecast origin, and closes at or before `s + H`                                                                                              | Rust `ForecastPlan::new`                                 | dynamic (constructor)                                   |
| I11 | Targets of one leaf do not overlap and are strictly increasing                                                                                                                         | Rust `ForecastPlan::new`                                 | dynamic                                                 |
| I12 | A target's support is the declared scientific period; a numerical approximation of it (snapping) never replaces it                                                                     | Rust alignment check, once the inference method is known | dynamic (refusal)                                       |
| I13 | A forcing evaluated outside its data domain is reported on every run; a data covariate read after the forecast origin is reported on every forecast; both are recorded in the artifact | Rust `RunNote` (every run) and `ForecastPlan` validation | dynamic (record, not refusal)                           |
| I14 | A rendered date parses back to the same model time                                                                                                                                     | Rust `ir::caltime` render/parse pair                     | dynamic (property test)                                 |

I1–I3 are compile-time and belong to the compiler because the source-language
semantics have one authoritative definition (§7). I4–I14 are runtime because
they concern data files, fits and forecasts, which the compiler never sees.

## 4. Types

Five runtime types are added or extended. Each is listed with the invalid state
it removes; the migration proposal (§5) gives the alternatives considered. Field
names are illustrative; the semantics are normative.

### 4.1 `ObservedRow` (extends the score-artifact proposal's type)

```rust
pub struct ObservedRow {
    pub position: usize,                 // index among this leaf's rows (§2.2)
    pub support: Support,                // the one temporal fact later code reads
    pub temporal: Vec<TemporalCell>,     // the cells as written, for the row digest only
    pub cell: Option<ObsCell>,           // None is a hole
    pub aux: Vec<(String, f64)>,
}

pub enum Support { Instant(f64), Period(Period) }   // Period is the existing type
impl Support { fn open(&self) -> f64; fn close(&self) -> f64; }
```

The score-artifact proposal's `ObservedRow` has `time: f64` "as the loader
resolves it" — the label. This design replaces it by `support`. The label is
still hashed (through `temporal`) but is not a field any decision can read.

_Prevents:_ comparing a label with a cutoff (the defect class behind gh#1062 and
the C2 counterexample, migration proposal §2.2). _Why existing types do not
suffice:_ `StreamTimes` holds supports per stream but is built after the holdout
cut, from rows whose only time is the label (`Observation.time`,
`particle_filter.rs:23`). _Constructed by:_ `observed_rows`, once per leaf.
_Consumed by:_ the likelihood loader, the partition, the row digest, the
forecast plan. _Serialization:_ not serialized; it is in-memory.

### 4.2 `TrainingPartition`

```rust
pub struct TrainingPartition {
    rule: PartitionRule,                 // how it was decided (provenance)
    extent: f64,                         // latest close over trained rows, all leaves
    leaves: Vec<LeafPrefix>,             // sorted by leaf name
}
pub struct LeafPrefix {
    pub leaf: String,
    pub n_trained: usize,                // rows [0, n_trained) trained
    pub digest: RowDigest,               // row digest v1 over those rows
}
pub enum PartitionRule {
    All,                                 // no holdout declared
    Cutoff { tau: f64 },                 // holdout_after = τ
    SeparateFiles,                       // [data.holdout]: training files are the prefix
}
impl TrainingPartition {
    pub fn from_cutoff(rows: &BTreeMap<String, Vec<ObservedRow>>, tau: f64)
        -> Result<Self, PartitionError>;
    pub fn whole(rows: &BTreeMap<String, Vec<ObservedRow>>) -> Self;
    pub fn contains(&self, leaf: &str, position: usize) -> bool;
}
```

Fields are private; the two constructors are the only producers. `from_cutoff`
trains row `i` of a leaf iff `close(support_i) ≤ τ`; because closes are strictly
increasing per leaf, the trained rows are a prefix (I8), and the constructor
asserts that rather than assuming it. A row that straddles `τ` is not trained
(§5.2).

_Prevents:_ (a) a fit that reports "training window [0, τ]" having trained on
events after `τ` (counterexample C2); (b) `holdout_after` and `[data.holdout]`
recording different things; (c) a later command re-deriving membership from
labels by a different convention (six sites today, migration §2.2). _Why
existing types do not suffice:_ `TrainingWindow { train_end: f64 }`
(`run_meta.rs:626`) is a bare label-axis number with no row count and no digest,
and is absent when no holdout is declared. _Serialization:_ written to
`fit.meta.json` as `training_partition`, replacing `training_window`:

```json
"training_partition": {
  "rule": {"kind": "cutoff", "tau": 24.0},
  "extent": {"t": 24.0, "date": {"value": "2026-07-25"}},
  "leaves": [{"leaf": "cases", "n_trained": 3, "digest": "9f2c…"}]
}
```

### 4.3 `ConditioningRecord`

```rust
pub struct ConditioningRecord {
    pub theta: TrainingPartition,        // Iθ, copied from the fit record
    pub state: StateConditioning,        // Ix
}
pub enum StateConditioning {
    None,                                // free-forward from t_start
    Filtered { through: f64, leaves: Vec<LeafPrefix> },  // rows assimilated up to the forecast origin
}
```

`Ix` is also a per-leaf prefix (a filter assimilates in time order), so it
reuses `LeafPrefix`. _Prevents:_ a band that conditioned the filter on held-out
rows being presented as a forecast (counterexample in migration §2.4:
`fit predict
--horizon one_step` filters through `holdout_after` rows with no
marking). _Serialization:_ in every predictive artifact and, for forecast
scores, in the score artifact beside `fit`.

### 4.4 `ForecastPlan`

```rust
pub struct ForecastPlan {
    mode: PredictiveMode,                // InSample | OneStepAhead | FixedOrigin
    origin: f64,                         // forecast origin s
    horizon: Option<f64>,                // H; None for OneStepAhead / InSample
    conditioning: ConditioningRecord,
    targets: Vec<Target>,                // sorted by (close, leaf)
    notes: Vec<PlanNote>,                // I13: forcing domain, future covariates
}
pub struct Target {
    pub leaf: String,
    pub support: Support,
    pub lead: Option<f64>,               // close − s under FixedOrigin
    pub observed: Option<usize>,         // row position when a held-out row exists
    pub source: TargetSource,            // HeldOutRow | DeclaredSpan | EmitSchedule
}
impl ForecastPlan {
    pub fn new(...) -> Result<Self, PlanError>;   // checks I10, I11
    pub fn required_times(&self) -> Vec<f64>;     // opens and closes, for snapshot recording
}
```

The plan is the seam between "what is to be predicted" (scientific, shared by
every backend) and "how it is computed" (per backend). Every predictive command
builds one plan and hands the engine `required_times()` plus, per target, the
`(open, close)` pair over which to difference cumulative flow — which is the
shape `fit predict` already uses (`leaf_row_coverages`, `predict.rs:3049`;
`required_output_times`, `predict.rs:1990-2105`). No engine changes its inner
loop.

_Prevents:_ three different target rules in four places (most-common observed
gap in `fit predict`, `emit_schedule` in `simulate --obs` and `--init-state`,
bound rows in `--design-from`; migration §2.3); forecast rows whose support is
wider than the observed rows' (window-column series); a straddling first period
dropped silently after a mid-period forecast origin.

### 4.5 `CalendarAxis`

```rust
pub struct CalendarAxis {                // in ir::caltime
    pub time_unit: TimeUnit,
    pub origin: Option<IsoDate>,         // None: unanchored
}
impl CalendarAxis {
    pub fn date_of(&self, t: f64) -> Option<RenderedInstant>;
    pub fn parse_instant(&self, s: &str) -> Result<f64, ParseError>;
}
```

Today the axis is reconstructed per consumer from the IR. One type gives every
artifact the same block (I4) and owns the render/parse pair (I14). It is the
score-artifact proposal's `TimeAxis.origin` and `time_unit` factored out so the
simulation leaf, the predictive artifacts and the score artifact share it.

### 4.6 What is deliberately not a type

- **No newtype per time role** (`ModelTime`, `LabelTime`, `ScoringTime`). The
  label/close confusion is structural — a row has both — and is removed by
  making the label unreadable (§4.1), not by wrapping `f64`. Inside the engine
  every time is model time, and the existing `OutputTimes` / `EffectTimes` /
  `ObsTimes` axis wrappers (`.claude/rules/rust-conventions.md`) already cover
  the swappable-vector case.
- **No observation identity beyond (leaf, position, prefix digest).** §10.
- **No IR-level instant/duration tags.** The compiler owns the distinction
  (I1–I3); the IR carries resolved numbers.

## 5. Observation membership and conditioning sets

### 5.1 Identity

A row is identified, within one fit, by `(leaf, position)`. A set of rows that
is a per-leaf prefix is identified by `(leaf, n, digest)`, where `digest` is the
score-artifact proposal's row digest v1 over rows `[0, n)`. Equality of two
prefixes means the same observations in the same order, independent of file
layout. This is sufficient because every set this design needs — `Iθ`, `Ix`, the
rows a score reads — is a prefix of some leaf's rows, and the held-out rows a
score scores are a contiguous range after `Iθ`'s prefix.

It is not sufficient for a revised surveillance series in which an early row's
value changed: the prefix digest changes and the score is refused as changed
data (the score-artifact proposal's `--allow-changed-data` override). That is
the right behaviour for this project; vintages are §10.

### 5.2 Cutoff as a policy

`holdout_after = τ` is a policy that constructs `Iθ`; it is not itself evidence
of membership. It is applied in two steps, both on supports:

1. **Training.** A row trains iff `close ≤ τ`. The training extent `E` is the
   latest close over the trained rows of every leaf, so `E ≤ τ`.
2. **Scoring.** Every untrained row is classified against `E`:

```
after(E)     open ≥ E          scorable as held-out; a forecast target
straddle(E)  open < E < close  neither trained nor scored
```

An instant support is never a straddler. A straddling row is assimilated by a
one-step-ahead filter (it is data, and the filter state must pass through it)
but is not scored, and is not a forecast target. The reason is that its period
contains time the parameters saw through another leaf; scoring it as held-out
would credit the model with predicting events the fit partly conditioned on
(counterexample C1: a weekly row `[21, 28)` scored as held out while daily rows
for days 21–24 trained the fit). Classifying against `E` rather than `τ` scores
every row whose period lies wholly after the data the parameters saw, which is
the largest honest held-out set; the two coincide whenever some leaf has a row
closing exactly at `τ`.

`[data.holdout]` (separate files) constructs `Iθ` as the whole of the training
files. The training extent is the latest training close. A holdout row must
satisfy `open ≥ extent`; a row that does not is refused at fit time, as today
(`runner.rs:2017`, which today compares labels).

### 5.3 What a fit records

Every fit records its `TrainingPartition` (I8), including fits with no holdout
(`PartitionRule::All`). The training extent is a close, so for a `day(time)`
stream whose last training label is 24 the extent is 25, and the fit's summary
says "trained on observations closing at or before 25 (2026-07-26)". The
partition rule version enters the fit's run identity (§8).

### 5.4 What a score proves

A score is stamped held-out iff (I9): for every scored leaf, the rows at
positions `[0, n_trained)` in the data the score reads digest to the fit's
`LeafPrefix.digest`; and every scored row is `after(extent)`. The first clause
is the score-artifact proposal's prefix check, restated on positions rather than
"rows at or before `train_end`" so that it does not depend on any time
comparison. A score never infers membership from a declaration.

### 5.5 Two information sets

`Iθ` and `Ix` are recorded separately because they legitimately differ:

1. One-step-ahead held-out scoring: `Ix` grows past `Iθ` one row at a time.
2. A forecast issued after new data arrived but before a refit: `Ix` extends to
   the latest data, `Iθ` stops at the last fit.
3. A free-forward projection: `Ix = ∅`.

A forecast artifact states both, in words: "parameters fitted to cases rows 1–30
(through 2026-08-01); state filtered through cases rows 1–33 (through
2026-08-22); forecast origin 2026-08-22".

## 6. Forecast-origin and target semantics

### 6.1 The forecast origin

Default: `s =` the latest close in `Ix`. With the default `Ix = Iθ`, this is the
training extent. A user-supplied `s` must satisfy `s ≥` the latest close in `Ix`
(no peeking) — a forecast origin earlier than data the state has seen is
refused. A forecast origin may fall inside a period of some leaf; that leaf's
straddling period is then not a target, and the plan records a note naming it
(today the row is dropped without notice, migration §2.3).

### 6.2 Where targets come from

For each leaf, targets after `s` are, in order of preference:

1. **Held-out rows** (`TargetSource::HeldOutRow`), when scoring: the supports of
   the rows `after(s)` in the data read. These are always the targets of a
   score; nothing is extrapolated.
2. **The declared span** (`TargetSource::DeclaredSpan`), for a leaf with a
   uniform `covers` form: periods tile forward from the last observed close with
   the declared span Δ: `[c, c+Δ), [c+Δ, c+2Δ), …`. The span is a model constant
   (`Covers::From/Until { span }`), so this is the stream's own cadence, not one
   inferred from data. An instant leaf with a regular `emit_schedule` uses that
   schedule.
3. **The emit schedule** (`TargetSource::EmitSchedule`), for a leaf whose
   `emit_schedule` is declared and whose cadence is not a uniform span (a
   window-column leaf with a regular or `at` schedule).
4. **Refusal** otherwise: a window-column leaf with no `emit_schedule` has no
   declared cadence, and the plan says so, naming the two ways to supply one.

The most-common observed gap is never a target source. The forecast horizon only
truncates: targets with `close ≤ s + H`.

Calendar-month cadences are not representable in an anchored model today (E322
refuses `every 1 'months`; E270 refuses `date_range` in
`emit_schedule at
[...]`; a month-wide `covers` width is refused once the
migration's Stage 0(c) lands). A monthly window-column leaf is therefore case 4
until the migration's calendar-period forms land (migration §6, Stage 7). With
them, a stream declared `covers = month(time)` (or an ISO/MMWR week form) is
case 2: its span is a calendar period, and targets tile forward by calendar
months (or weeks) from its last close. Refusing is correct in the meantime: a
30-day step is a different target.

### 6.3 Accumulation

A flow target `[a, b)` is predicted as `N(b) − N(a)` of the stream's cumulative
projection, with `a` and `b` both recorded snapshot times (`required_times`).
This is exact on every backend at the snapshot times; when `a < s` the target
does not exist (§6.1). For the ODE skeleton the difference is of the integrated
flow; for the chain-binomial and Gillespie backends it is of the counting
process; the scientific target is identical and the numbers differ by the
backends' legitimate differences.

### 6.4 Alignment

The scientific support of a target or observation is fixed by §2.2. A backend
that can evaluate only on its step grid evaluates an approximation. There is one
rule, determined by what the inference method can do, measured from `t_start`:

- A method that can step exactly (if2, pfilter, pmmh) steps to every open and
  close. No approximation.
- A method that cannot (PGAS) requires every open and close to lie on the grid
  `t_start + k·dt` within the named grid tolerance. If one does not, the run is
  refused, naming the row and the nearest grid times. Snapping is never used to
  move a support to a different period (I12).

This is a capability of the method, not a user setting. Under it a user-facing
alignment policy changes no number, so the `[config] obs_alignment` key is
removed (migration Stage 3). A policy-blind `time % dt` check before the method
is known is removed.

### 6.5 Forcings and covariates

Each forcing with a data domain (interpolated, piecewise, cubic, B-spline: the
constant `times`/`breakpoints` in the IR) has a domain `[first, last]`. Two
notes (I13) are recorded, never a refusal:

- **Held past its domain**, on **every run** — `simulate`, `fit`, `pfilter`,
  every predictive and scoring command — not only on forecasts: the run
  evaluates the forcing at `t > last` (or `t < first`), where it holds its
  endpoint value. The note is a `RunNote`, said once per forcing, naming the
  knot range and the span held. Today only B-splines report this
  (`spline-held-past-window`, `obs_anchor.rs:497-560`). A forecast carries the
  same note into its artifact.
- **Read after the forecast origin:** the forcing's domain extends past `s`, so
  the forecast reads covariate values that would not have been available at `s`
  if they are observations rather than a scenario. camdl cannot know which; the
  note names the forcing and the span, and the artifact records it, so a reader
  can tell a scenario forecast from a leaked one.

Endpoint holding is a legitimate scenario assumption, so it is stated rather
than refused. This overrides the refusal proposed in
`docs/dev/proposals/2026-06-09-time-interval-model.md` §7.1, point 5. An
explicit per-forcing extrapolation declaration, which would make a refusal
available per forcing, is deferred (§10).

## 7. Compiler / IR / runtime boundary

The path is: DSL source → parse → time typing (`time_typing.ml`) → constant
evaluation and expansion (`expander.ml`) → typed IR → JSON → Rust `ir` →
observation loading (`observed_rows`) → partition and plan → engines →
artifacts.

- **The compiler owns the source-language semantics of time**: instant vs
  duration, exact vs calendar, `date()` meaning the start of its day, calendar
  stepping. It lowers every temporal expression to model time `f64` in the IR.
  This erasure is correct and stays: the engine has no use for the distinction,
  and a second time language in Rust would be two definitions of one semantics.
  The compiler's defects are in that lowering (I1–I3 holes; migration §2.1), not
  in its types.
- **The IR carries what the runtime needs to reconstruct temporal meaning of
  data**, and already does: the calendar axis (`time_unit`, `origin`,
  `origin_rata_die`), each stream's `covers` (offset and span, or window
  columns), column roles, emit schedules, forcing knot times. No new IR field is
  required for Stages 1–6. One IR change is proposed for calendar periods, Stage
  7: a calendar `covers` variant (`month(time)`, ISO and MMWR weeks) and a
  calendar-month emit schedule for instant streams.
- **The runtime owns everything about data, fits and forecasts**: supports,
  partitions, conditioning sets, plans, alignment, notes, artifacts. Every
  runtime temporal type is constructed at one boundary and the engines receive
  `f64` vectors (`Schedule`, `StreamTimes::closes`, `required_times`).
- **The runtime never re-interprets source syntax.** The CLI time grammar
  (`parse_time_spec`, `runner.rs:2162`) is the runtime's one parser for times
  typed by a user; durations on the CLI (`--horizon "4 weeks"`) go through it.
  Its month and year are the affine `365.2425/12`-day spans, as the IR's are; a
  calendar-month horizon is not accepted on the CLI, matching E321.

## 8. Artifact contracts

Every artifact that carries a model time carries the model's `CalendarAxis`
once, and every instant it names as `{t, date}` with `date` a `Recorded` string
(score-artifact proposal's `TimePoint`). Specifically:

- **Simulation leaf** (`run.json`): gains `calendar`. Today it has none
  (gh#838), so a stored trajectory cannot be dated from its leaf.
- **Fit record** (`fit.meta.json`): `training_partition` (§4.2) replaces
  `training_window`; `calendar`.
- **Predictive artifacts** (`fit predict`, `simulate --init-state`): the plan
  (`mode`, forecast origin, horizon, conditioning record, notes) in the JSON
  sidecar; per row in the TSV, `role ∈ {trained, held_out, straddles, target}`
  and the support (`t_start`, `t_stop`, plus dates when anchored), matching the
  trajectory tables' existing `t_start`/`t_stop` columns.
- **Score artifact**: `ScoredLeadTime` gains `FromOrigin { origin: TimePoint }`,
  a minor version of `camdl.score` as that proposal anticipates; each scored
  target records `lead`; scores at different leads are stored per lead and never
  summed (`2026-08-29` §3.7.2). `fit` gains the `ConditioningRecord`.

Dates are rendered and parsed by one pair in `ir::caltime` (I14). A sub-day time
renders in a form the loader accepts.

**Identity.** The partition rule is identity: a fit under `holdout_after = τ`
trains on different rows after this change for non-`closing_at` streams, so the
change rides a `HASH_VERSION` bump, the same one the score-artifact proposal
schedules (migration §7). The forecast plan's mode, origin, horizon and target
list are identity of any stored predictive or score leaf. Notes are
presentation.

## 9. Names

| Concept                                        | Name                       | Retired spellings                                                                                                                |
| ---------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `t = 0` calendar instant                       | origin                     | —                                                                                                                                |
| instant predictive dynamics start free-running | forecast origin            | "origin" alone                                                                                                                   |
| latest close over `Iθ`                         | training extent (`extent`) | `train_end` as a label time                                                                                                      |
| boundary instant of an information set         | conditioning cutoff (`τ`)  | "cutoff", `score_from` as a user concept                                                                                         |
| maximum lead time                              | forecast horizon           | `fit predict --horizon` as a mode; `compare --allow-mismatched-horizon` as a scored-step count; `simulate --to` help's "horizon" |
| close − forecast origin                        | lead time                  | "horizon k"                                                                                                                      |
| instant a row's information is complete        | close                      | "scoring time" (kept as a synonym in the score page: "scored at")                                                                |

The published-comparison principles bar the word "horizon"
(`2026-11-13-published-comparison-principles.md`, "Vocabulary") because it had
four meanings. This design gives it exactly one and frees the others; the
principles' vocabulary needs the matching amendment (migration, decision D3).

## 10. Deferred, and why the first implementation does not block it

| Generalization                         | What it would add                                                        | Why nothing here must be rewritten                                                                                                   |
| -------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| Observation identity across datasets   | a stable key per row independent of position (e.g. leaf label + support) | `LeafPrefix` is the only identity consumers read; a key-based set replaces its `n_trained` with a key set behind the same `contains` |
| Data vintages and `as_of` queries      | per-row revision history; "the data as known at `s`"                     | `ConditioningRecord` already separates `Ix` from `Iθ`; a vintage becomes one more field of `LeafPrefix` (which file version)         |
| Retrospective real-time reconstruction | rebuilding `Ix` from vintages                                            | a `ForecastPlan` built from a vintage-selected row set; the plan is unchanged                                                        |
| Rolling-origin evaluation              | `J` forecast origins                                                     | `J` plans over `J` partitions; scores are already per lead and per origin (`ScoredLeadTime::FromOrigin`)                             |
| Hubverse import/export                 | external quantile/sample forecasts                                       | a `Target` is the hub's (target, `target_end_date`) join key; the score artifact's 23 quantile levels are the hub's set              |
| Explicit forcing extrapolation         | `extrapolation = hold \| error` per forcing                              | I13's notes become the default for `hold`; `error` is a new refusal                                                                  |
| ISO/MMWR week `covers` forms           | week phase from a standard, checked at load                              | planned in the migration's Stage 7 as the same IR variant as `month(time)`; a calendar span is one more case-2 target source (§6.2)  |
| Comprehensive provenance graphs        | lineage of every artifact                                                | the score artifact's `ArtifactRef` lineage already exists                                                                            |

## 11. Worked examples

**E1. Day-labelled stream across a cutoff.** `covers = day(time)`, labels
`1..16`, `holdout_after = 10`. Supports `[d, d+1)`. Row 10 closes at 11 > 10, so
it is not trained (today it is: the label 10 ≤ 10). Training extent = 10 (row
9's close). Held-out rows: labels 10..16 (opens ≥ 10). The fit says "trained on
9 rows closing at or before 10".

**E2. Two cadences.** `weekly`: `closing_at(time, 7 'days)`, labels 7..70.
`daily`: `day(time)`, labels 0..69. `τ = 24`. Daily rows 0–23 train (closes
1..24). Weekly rows 7, 14, 21 train (closes ≤ 24). Extent `E = 24`. Weekly row
28 covers `[21, 28)`: straddles `E`, not scored. Weekly row 35 `[28, 35)`:
scored. Daily row 24 `[24, 25)`: scored. Compare today scores weekly row 28 as
held out.

**E3. Monthly window columns, forecast.** Window rows close at 31, 59, 90, 120,
151, 181 (calendar month-ends). No `emit_schedule`. A forecast with
`H = 90
'days` is refused: "stream 'cases' declares its periods per row and has
no `emit_schedule`; forecast targets need a declared cadence". Today
`fit predict` continues a 30-day step: 211, 241, … where month-ends are 212,
243, ….

**E4. Forecast origin mid-period.** Weekly `closing_at(time, 7 'days)`, state
filtered through the row closing at 25. With the default forecast origin 25,
targets tile from the last close: `[25, 32)`, `[32, 39)`, …. With a requested
forecast origin of 27 (later than `Ix`'s last close, so allowed), the period
`[25, 32)` straddles 27: it is omitted with a note naming it, and the first
target is `[32, 39)`. Today `simulate --init-state` drops such a row without
notice.

**E5. Invalid: forecast origin before conditioning data.** `Ix` through close
40, requested `s = 35`: refused, "the forecast origin 35 is earlier than data
the state was conditioned on (close 40)".

**E6. Invalid: snapped support.** PGAS (`Snap`), `t_start = 0.5`, `dt = 1`,
observations at labels 3, 5, 7 under `closing_at(time, 2 'days)`: supports
`[1, 3)`, … are not on the grid `0.5 + k`. Refused, naming row 3 and grid times
2.5 and 3.5. Today PGAS scores `[1.5, 3.5)` and labels the row 3.

**E7. Covariate after the forecast origin.** An interpolated forcing with knots
through `t = 100`, forecast origin 60: the plan's note "forcing 'temp' reads
data values after the forecast origin over (60, 100]; if these are observed
covariates, the forecast uses information not available at the forecast origin".
A forecast to 120 adds "forcing 'temp' holds its last value over (100, 120]".
