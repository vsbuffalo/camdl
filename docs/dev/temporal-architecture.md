# Temporal architecture: observation periods, training data and forecasts

**Status:** Normative. Decisions are cited by their number in the maintainer's
docket (D35 onward). The staged implementation plan is
`docs/dev/proposals/2026-11-15-temporal-architecture-migration.md` (the
migration plan); the `camdl predict` and `camdl score` commands are
`docs/dev/proposals/2026-11-23-prediction-and-scores.md`.

**Reader.** A contributor who needs to know how camdl represents time from a
data file through a fit to a forecast and its score, and which code owns each
rule. The terms for time are defined below, under "Terms for time". The
modeler's account of calendar time, with the rules for writing dates in a model,
is `docs/dates.md`.

## Summary

This document covers how a row of a data file is given a time span, how a fit
records which rows it trained on, how a fixed-origin forecast chooses what to
predict, and which code enforces each rule. The rest of the document relies on
these facts:

- Each data row measures an instant or a period, computed once from the row's
  time column and its stream's `covers` declaration. Training, scoring and
  forecasting read that span. They do not read the date written in the time
  column.
- A fit records its training rows as the first _n_ rows of each stream, with a
  digest of those rows. `camdl predict` and `camdl score` read that record and
  refuse data whose leading rows no longer match it.
- A fixed-origin forecast starts at the end of the last training row. It
  predicts a list of targets, each a stream and a period or instant, which the
  forecast plan builds before any simulation runs.
- A fixed-origin forecast exists on the chain-binomial backend only (§2.4).
- The simulation engines work in plain floating-point times. The types described
  here sit between the compiled model (the IR) and the engines.

## Terms for time

The examples in this section use a model that declares `time_unit = 'days` and
`origin = date("2026-07-01")`.

**Model time**, written `t`, is a real number: the count of time units since
`t = 0`. Every time the runtime handles is a model time. In the example model,
`t = 2` is the start of 3 July and `t = 2.5` is noon on 3 July.

**`time_unit`** is the model's declaration of what one unit of `t` is: days,
weeks, months or years. Under `time_unit = 'weeks` the start of 8 July would be
`t = 1`.

**`origin`** is the model's optional declaration of the calendar date at
`t = 0`. With it, a date converts to a model time as the number of days from the
origin to the date, divided by the days in one time unit, so the start of 5 July
is `t = 4`. A model without an `origin` has no calendar: its times are bare
numbers, and its artifacts carry no dates.

An **instant** is one point on the time axis, a single value of `t`. A date
names the instant at which its day begins: `date("2026-07-05")` is `t = 4`, and
is not the whole of 5 July.

A **duration** is a length of time with no position on the axis, such as
`7 'days`. The difference between two instants is a duration.

A **period** is the stretch of time between two instants, its start and its
stop. It contains its start and not its stop, written `[start, stop)`. The day 5
July is the period `[4, 5)`, and the week of 5 to 11 July is `[4, 11)`.

`docs/dates.md` gives the full account, including calendar months and years,
whose lengths vary.

## Part 1. Time below the compiler

### 1.1 The calendar is erased at the compiler and loader

The compiler checks the source-language rules for time (an instant is not a
duration; a calendar month cannot shift an instant except through
`add_calendar_months`; `date("2026-07-05")` is the instant 5 July begins) and
lowers every temporal expression to model time, a number in the model's
`time_unit` counted from `origin`. The data loader converts dates in data files
the same way. Below those two places every time is an `f64` (`docs/dates.md`,
"The one rule"). The engines have no use for calendar structure, and a second
time language in Rust would be a second definition of one semantics.

The runtime's primitive for a data row's time span is `Period`
(`rust/crates/sim/src/inference/multi_stream_obs.rs`), a half-open interval
whose only constructor refuses a non-finite or non-increasing pair, and
`StreamTimes`, which holds a stream's instants or periods. `Covers::period_of`
(`rust/crates/ir/src/observation.rs`) turns a label into a period under each
uniform `covers` form.

## Part 2. Concepts

### 2.1 The time axis

Every time in the runtime is a model time `t`, in the model's `time_unit`
("Terms for time", above). `origin` only says which calendar date is `t = 0`;
changing it renumbers every time and moves nothing in calendar time. It is
unrelated to the forecast origin (§2.5), and the word "origin" alone always
means the calendar one.

### 2.2 What a data row measures

A data row holds one observed value of one stream. The value describes one of
two things:

- an **instant**, when the row reads the state at one moment, such as a census
  of occupied beds (`projected = I`);
- a **period**, when the row counts events accumulated over a stretch of time,
  such as the cases reported in a week (`incidence(...)`).

The instant or period is what the row **measures**. In the code it is the row's
`Measures` (§3.1), either `Instant(t)` or `Period([start, stop))`.

The **end** of what a row measures is the instant for an instant row and the
period's stop for a period row. It is when the row's information is complete,
and it is the time at which the row is scored (`StreamTimes::closes`).

The **label** is what the file writes in the row's time column, a number or a
date. A label names a row; alone it does not say what the row measures. For an
instant row, the label is the instant. For a period row, the stream's `covers`
declaration says which period the label names:

| `covers` form                  | label 21 means        | the period's end |
| ------------------------------ | --------------------- | ---------------- |
| `day(time)`                    | day 21                | 22               |
| `starting_on(time, 7 'days)`   | days 21–27            | 28               |
| `ending_on(time, 7 'days)`     | days 15–21            | 22               |
| `closing_at(time, 7 'days)`    | days 14–20            | 21               |
| `window_start` / `window_stop` | read from two columns | `window_stop`    |

The same label names four different periods under the four uniform forms, so a
rule that compared labels with a cutoff would give different answers for the
same data. Training, scoring and forecasting therefore read what the row
measures, or its end. The label is kept for the places where the file's own time
is the answer, such as writing the time column back to a file (§3.1), and it is
hashed into the row digest (§3.2).

A row's **position** is its index among its stream's rows in file order. The
binder requires each stream's ends to be strictly increasing (`BoundObs::bind`),
so position order is time order. A uniform `covers` form also forbids gaps
(`pfilter::stream_times_of_measures`: a gap is a missing row; an unobserved
period is written as an `NA` row), so a uniform stream's periods tile without
gaps or overlaps.

A **stream** here is one concrete observation series. The code calls it a _leaf_
when a declared block expands into several (an indexed block over two patches is
two leaves); every statement below about a stream applies per leaf.

### 2.3 Training data and the cutoff

A fit config may declare `holdout_after` under `[data]`:

```toml
[data]
holdout_after = "2026-07-25"   # or 24 (model time, in time_unit), or "last_obs - 6 weeks"
```

The value is an instant, the **cutoff**. A date means the instant that day
begins, so `"2026-07-25"` is the start of 25 July. A row **trains** if the end
of what it measures is at or before the cutoff (§4.1). The **training extent**
is the latest end among the training rows of every stream; it is never later
than the cutoff, and it is what the fit records and reports as "trained
through".

A fit may instead name separate holdout files (`[data.holdout]`). Then every row
of the training files trains, and the training extent is again the latest end
among them.

Rows that did not train are **held out**. A held-out score scores every held-out
row whose end is after the training extent (§4.2).

### 2.4 Two information sets

A prediction is computed from two quantities estimated from data: the
parameters, and the latent state at the time the prediction starts. Each was
informed by a set of data rows. An information set, in this document, is such a
set of rows, recorded per stream as a row count and a digest (`RowPrefix`,
§3.2).

- The **parameter set** is the rows the parameters were fitted to. These are the
  training rows (§2.3).
- The **state set** is the rows the latent state at the forecast origin (§2.5)
  was conditioned on. To condition the state on rows is to run a particle filter
  through them, so that the particles at the origin approximate the distribution
  of the state given those rows' values.

For a fixed-origin forecast the two sets hold the same rows. The forecast origin
is the training extent, and the forecast driver filters through exactly the
training rows (D57; `forecast_driver::conditioning`). `forecast.json` records
them separately, as `parameter_set` and `state_set`, because they can differ in
principle: conditioning the state on rows newer than the fit, without a refit,
would make the state set the larger one. D35 allows that and rolling forecast
dates need it; it is unsupported (Part 7).

The other two modes of `camdl predict` (§2.6) have different state sets, which
`predictive.json` records per mode as `state_sets`. A free-forward projection
runs from the start of the run and its state set is empty. Its parameters were
still fitted to the training rows, so it is conditioned on the data through the
parameters. A one-step band at a row is drawn from a state filtered through
every earlier row of every stream, held-out rows included.

Two properties of a fit decide what a forecast of it is and whether one exists:
the simulation backend and the inference method. They are separate. PMMH, PGAS
and IF2 are three inference methods that all simulate the model on the
chain-binomial backend.

**The simulation backend** decides whether the state is random once the
parameters are fixed, and so whether conditioning it on rows changes anything.

- **Chain-binomial** simulates the model in discrete steps with random
  transition counts, so the state is random given the parameters. A particle
  filter run through the rows up to the forecast origin leaves particles that
  approximate the state's distribution given those rows, and the forecast
  continues each particle. Each particle also carries, per stream, the flow
  accumulated since that stream's current period began (`ParticleState::acc`,
  `rust/crates/sim/src/inference/types.rs`, copied with the particle on
  resampling), which is what makes a period already under way at the origin
  predictable (§4.4). This is the only backend with a fixed-origin forecast.
- **ODE** integrates the model deterministically, so the state is fully
  determined by the parameters and conditioning it on rows changes nothing. The
  data reach an ODE prediction only through the parameters, and its state set is
  always empty. Rolling ODE forecasts with fixed parameters would therefore be
  slices of one trajectory and would not use newer data, while a stochastic
  model at the same forecast dates would. A fixed-origin forecast on the ODE
  backend is refused (`fit::forecast_output::refuse_unsupported`); an ODE fit is
  predicted with `--mode free_forward`.
- **Gillespie** simulates the model event by event. It is not a fitting backend
  and runs no particle filter, so a forecast on it is refused.

**The inference method** decides how the parameters enter the forecast.

- **PMMH and PGAS** produce posterior draws. The forecast runs one particle
  filter per draw and pools the results, so its bands carry parameter
  uncertainty. `forecast.json` labels it `"treatment": "posterior"`. PGAS also
  stores sampled state paths, and the forecast does not use them: a path is a
  draw given all of the fit's data, so it is a valid state at the origin only
  when the origin is the end of that data. For an earlier origin, later data
  shaped it; a later origin needs the newer rows filtered, which is unsupported
  (D57).
- **IF2** produces one point, the run's winning parameter values. The forecast
  runs one particle filter at that point, labelled `"treatment": "plug_in"`. Its
  bands carry process noise and observation noise and no parameter uncertainty.
- **MH and the NLopt optimizers** fit on the ODE backend, so their fits are
  refused a fixed-origin forecast whatever the method produced.

| Inference method                    | Backend        | Parameters enter as | Fixed-origin forecast |
| ----------------------------------- | -------------- | ------------------- | --------------------- |
| PMMH, PGAS                          | chain-binomial | posterior draws     | yes, `posterior`      |
| IF2                                 | chain-binomial | one winning point   | yes, `plug_in`        |
| MH                                  | ODE            | posterior draws     | refused (backend)     |
| `nl-sbplx`, `nl-bobyqa`, `nl-lbfgs` | ODE            | one winning point   | refused (backend)     |

A parameter point given in a file, with no fit behind it, is forecast on the
chain-binomial backend in the same way as an IF2 winner, pooled over `--n-draws`
copies of the point that differ only in their random seeds. The supported and
refused cells are tested in `rust/crates/cli/tests/fit_predict_forecast.rs`.

### 2.5 Forecast origin, targets, horizon and lead time

The **forecast origin** is the instant after which a forecast sees no data. It
is the training extent (§2.3), which is also the latest end among the rows of
the state set (§2.4). `ForecastPlan::new` takes no origin; it reads the extent
of the training partition. Setting a different forecast origin is unsupported
(Part 7).

A **target** is one stream's period (or instant) that the forecast predicts. A
target exists whether or not an observation for it exists; when one does, the
target carries the observed row's position.

Four quantities describe a target, and they are kept separate:

- the **forecast origin**, any instant; it is not moved onto any stream's
  reporting schedule;
- the **target interval**, the stream and the period (or instant) predicted;
- the **lead time**, the duration from the forecast origin to the end of the
  target interval;
- the **horizon**, an integer: the position of the target among its stream's
  targets, counting the first target that ends after the forecast origin as
  horizon 0. For a stream whose periods tile, this is the number of the stream's
  own periods between the period containing the origin and the target's period;
  the same rule also covers instant streams and window columns with gaps. A
  stream's unit is its declared width or schedule, so a weekly stream counts
  weeks and a daily stream counts days, with no separate setting.

Forecast hubs count horizons the same way in structure: in the CDC COVID-19 hub
and FluSight, `target_end_date = reference_date + horizon × 7 days`, with the
unit set per target (the Reich-lab COVID hub counted "N wk ahead" and "N day
ahead" targets side by side). Their horizon 0 is the week that begins the day
after the data stop (the COVID hub's example: data through 2025-06-07, horizon 0
the week of 8–14 June), and the European hub's horizon 0 is instead the last
completed week. A hub's horizon is therefore not camdl's index: hub export
computes it from each target's end date and the hub's reference date, which the
hubs say is sufficient to align forecasts with observations.

On the command line, `--horizon "4 weeks"` is a duration: the forecast includes
every target whose period ends within 4 weeks of the forecast origin. With an
origin on a week boundary, a weekly stream gets horizons 0–3 and a daily stream
horizons 0–27. The two uses of the word (the count on a target, the reach on the
command line) follow the literature, which uses "horizon" for both (Hyndman and
Athanasopoulos, _Forecasting: Principles and Practice_, 3rd ed.).

![Figure 1: a weekly stream; the parameters were fitted to the first six rows and the state was filtered through the same six; the forecast origin is the end of the sixth period; the next four periods are targets with horizons 0 to 3, each with a lead time from the origin to its end.](figures/temporal-information-sets.svg)

_Figure 1. The two information sets and one forecast._ The parameter set and the
state set are both the training rows, and the forecast origin is the training
extent.

When streams report at different rates, or the forecast origin falls inside a
period, a stream's current period has begun before the origin. It is that
stream's horizon 0 and is predicted (D39): the part of the period before the
origin is inferred from the state, and the part after it is predicted (§4.4).
For a stream on a single cadence with the origin at the end of its data, the
first target is a whole period, as in the hubs.

![Figure 2: a daily stream observed through day 23 and a weekly stream whose last row covered days 14 to 20; the forecast origin is the start of day 24; the daily targets are days 24, 25, 26 with horizons 0, 1, 2; the weekly week of days 21 to 27 is horizon 0, with days 21 to 23 inferred and days 24 to 27 predicted.](figures/temporal-two-cadences.svg)

_Figure 2. Horizons on two streams that report at different rates._ Each stream
counts horizons in its own period; the weekly stream's week under way is its
horizon 0.

### 2.6 The axes of a prediction

A prediction is described by four properties. Each is recorded under its own
name.

**Mode: which rows the state was conditioned on.** The mode is the value of
`camdl predict --mode` (`PredictModeArg`; D50), and `--horizon` alone selects
`forecast`.

- `free_forward`: none. The model runs from the start of the run.
- `one_step`: at each data row, every earlier row of every stream. Its lead is
  the time since the last assimilated row of any stream, which varies.
- `forecast`: the training rows up to the forecast origin, and no row after it.

The mode is the `mode` column of the predictive, forecast and score tables.
`predictive.json` records each mode's state set under `state_sets`.
`forecast.json` records `"mode": "fixed_origin"`. The forecast plan has no mode
field, because only a fixed-origin forecast builds a plan.

**Whether the parameters were fitted to the predicted row.** This is a property
of each row and of each score, independent of the mode: a `one_step` band covers
training rows and held-out rows alike. "In-sample" is a value here, not a mode.
Per row it is the `role` column (`RowRole`): `trained`, `held_out`, `future` for
a period with no data row, or `unknown` for a parameter point with no record of
its training rows. Per score it is `conditioning` in `score.json`
(`score::trace::Conditioning`): `in_sample`; `hold_out_tail`, when the fit's
recorded partition shows the scored rows were not trained on;
`hold_out_declared`, when a parameter point's training rows were stated with
`--holdout-after` and cannot be verified; or `unknown`.

**Parameter uncertainty.** A prediction either pools over posterior draws or
uses one point (§2.4). It is `treatment` in `forecast.json` and the predictive
tables (`TreatmentKind`: `posterior` or `plug_in`), and `provenance` per mode in
`score.json` (`score::trace::Provenance`).

**Backend.** Chain-binomial, ODE or Gillespie (§2.4). The backend limits the
mode: `one_step` and `forecast` need the particle filter and so the
chain-binomial backend, and `free_forward` runs on all three. A score records
the numerical method that results, as `basis` per mode in `score.json`
(`score::trace::PredictiveBasis`): `filtered` for `one_step` and `forecast`,
`ode_free_forward` for a free-forward ODE trajectory, and
`simulated_free_forward` for a free-forward stochastic simulation.

The first three properties vary independently of one another. The backend is
independent of the second and third and constrains the first as stated.

## Part 3. Type design

![Figure 3: the compiler and IR carry resolved numbers; in the runtime, one row reader produces rows with their periods, the training partition is built from them and written to the fit record, the forecast plan reads the partition and the filter's record of what it assimilated, and the engines receive plain numbers.](figures/temporal-layers.svg)

_Figure 3. Layers, seams and ownership._ The temporal types form a thin band
between the IR and the engines. Each is built at one boundary; the engines see
only `f64` vectors through the existing `Schedule` and `StreamTimes::closes`
seams.

### From a data file to a score

This section follows one forecast from the data file to its score and names the
type at each step, with the invalid state that type or its constructor rules
out. Where a rule is held by a run-time check or by a convention between a
function and its caller, and not by the type, the text says so. Sections 3.1 to
3.4 then describe each type.

![Figure 4: six steps from top to bottom: the row reader produces ObservedRow values with what each measures; the fit builds a TrainingPartition, a row count and digest per stream, and records it in fit.meta.json; later commands read it back checked against the data; ForecastPlan takes its origin from the partition and lists the targets; the forecast driver runs a particle filter per parameter draw; camdl score scores each target against an outcome row. Beside each step is the invalid state it rules out.](figures/temporal-type-flow.svg)

_Figure 4. The types a forecast passes through._ Each arrow is labelled with the
function that builds the next value.

**Rows.** `observed_rows` reads each stream's file once and returns one
`ObservedRow` per row, with its `Measures` computed from the label and the
stream's `covers` declaration. Every later step works from these rows, not from
the file. The row keeps its `label` as a readable field, so nothing in the type
stops a later step from comparing a label with a cutoff. The rule is held by one
function, `training_partition::trains_under_cutoff`, which reads
`measures.end()`.

**The training partition.** A fit builds a `TrainingPartition` from the rows.
For each stream it holds a `RowPrefix`: a count `rows` and a digest of the first
`rows` rows. A row trained exactly when its position is below the count. Within
one file, therefore, a row cannot be both trained and held out, and the training
rows cannot have a gap. The fields are private and the constructors are
`from_cutoff`, `whole` and `from_record`. `from_cutoff` checks at run time that
the rows ending at or before the cutoff are a prefix (`NotAPrefix` otherwise),
and refuses a split that trains nothing in some stream or holds nothing out.
Under `[data.holdout]` the held-out rows are in separate files that the
partition does not describe; a run-time check, `runner::check_holdout_files`,
refuses a holdout row that ends at or before the training extent.

**The record.** The fit writes the partition to `fit.meta.json`. A later command
does not apply the cutoff again. `TrainingPartition::read` takes each stream's
count from the record, digests the first `rows` rows of the data it has just
read, and refuses the data if the digest differs, if the file now holds fewer
rows, or if the record was written under another version of the rule. A later
step therefore cannot arrive at a different split, and cannot use a fitted row
that was edited after the fit; rows appended after the prefix do not affect the
check. This is a run-time check, and it is as strong as the digest. The stored
fit keeps each held-out row's file, position and time span and not its value
(`forecast_output::held_out_rows` reads them with an empty cell), so
`camdl predict` has no held-out value to read.

**The plan.** `ForecastPlan::new` takes the verified partition, the reach and
each stream's rows. It takes no origin: the origin is `partition.extent()`, so a
plan cannot start anywhere but the end of the training rows. Its fields are
private and `new` is the only constructor. Targets are made in one function,
`stream_targets`, which keeps only those ending after the origin, and `new`
asserts that and the ordering of each stream's targets. A target's `observed`
field, the held-out row that measures the same period, is filled only from the
rows the caller passed as held out, and only from those ending after the origin.
Every training row ends at or before the origin, so `observed` cannot name a
training row. That the rows passed as held out lie after the training prefix is
the caller's convention; the plan does not check it against the partition.

**The driver.** `fixed_origin_forecast` takes the plan and the streams' training
rows. Before filtering, `conditioning` digests the rows it was handed and fails
unless each stream's count and digest equal the plan's parameter set and every
row ends at or before the origin. That function is the only place a
`StateConditioning` is built, so the recorded state set describes the rows the
filter was given. The targets are then appended to their streams as rows with no
value, or with the held-out value wrapped as `ObsCell::HeldOut`. `ObsCell` has
two variants: the filter assimilates a `Scalar`, and a `HeldOut` is read only by
the scoring likelihood. The driver has no path that makes a `Scalar` for a
target, and `declared_read` refuses to read a prediction at a stop that holds an
assimilated value.

**The score.** `camdl score` is given an outcome file per stream.
`score_command::check_agreement` compares it row by row, by digest, with the
rows the prediction conditioned on, and refuses a file that does not begin with
them. Only the rows after those (`OutcomeBinding::after_conditioning`) can
supply a target's value, matched to the target by its period, and two rows that
measure one target are refused. The score is labelled with its `conditioning`
(§2.6), which is `hold_out_tail` only when the fit's recorded partition shows
the scored rows were not trained on.

Four runtime types carry the design. For each: what it represents, the invalid
state it removes, who builds it, who reads it, and how it is stored. Field names
are illustrative; the meanings are normative.

### 3.1 `ObservedRow`: one data row, with what it measures

```rust
// rust/crates/cli/src/fit/observed_rows.rs (crate-private)
pub(crate) struct ObservedRow {
    pub position: usize,             // index among this stream's rows (§2.2)
    pub label: f64,                  // the row's time cell in model time; never compared
    pub measures: Measures,          // the one temporal fact later code reads
    pub temporal: Vec<TemporalCell>, // the time cells as written; hashed, never compared
    pub cell: Option<ObsCell>,       // None is an NA row
    pub aux: Vec<(String, f64)>,     // e.g. a binomial denominator
}

pub(crate) enum Measures { Instant(f64), Period(Period) }  // Period is the existing type
impl Measures { pub fn start(&self) -> f64; pub fn end(&self) -> f64; }
```

- **Removes:** any comparison of a label with a cutoff. `label` is a plain
  field, the time cell in model time, and is read only where the file's own time
  is the answer: writing the time column back to a file, the time column of
  `camdl predict`'s tables, naming the row in a refusal, and the check that no
  row is labelled before the run starts. No training, scoring or anchoring
  decision reads it; those read `measures`.
- **Why the existing types do not suffice:** `StreamTimes` holds periods per
  stream but is built after the training cut, from rows whose only time is the
  label.
- **Built by:** `observed_rows(...)`, the one reader, once per leaf; it is the
  score-artifact proposal's `observed_rows` with `measures` in place of its
  `time`. `measures_of` is the one lowering from labels to what rows measure; a
  stream with no `covers` declaration reads each label as an instant, and an
  accumulating stream with no declaration is an IR the compiler cannot produce
  (E350).
- **Read by:** the likelihood loader (`runner::load_observations`), the stream's
  periods (`pfilter::stream_times_of_rows`), the training partition, the row
  digest, and the forecast plan (`forecast_plan::PlanStream`).
- **Stored:** in memory only.

### 3.2 `TrainingPartition`: which rows the parameters were fitted to

```rust
pub struct TrainingPartition {
    rule: PartitionRule,         // provenance: how it was decided
    extent: f64,                 // latest end among training rows, all streams
    streams: Vec<RowPrefix>,     // sorted by stream name
}
pub struct RowPrefix {
    pub stream: String,
    pub rows: usize,             // rows [0, rows) of this stream
    pub digest: RowDigest,       // row digest over exactly those rows
}
pub enum PartitionRule { All, Cutoff { at: f64 }, SeparateFiles }

impl TrainingPartition {
    pub fn from_cutoff(rows: &BTreeMap<String, Vec<ObservedRow>>, at: f64)
        -> Result<Self, PartitionError>;
    pub fn whole(rows: &BTreeMap<String, Vec<ObservedRow>>, files: WholeFiles)
        -> Result<Self, PartitionError>;
    pub fn rows(&self, stream: &str) -> usize;   // 0 for a stream the fit did not bind
    pub fn rule(&self) -> PartitionRule;
    pub fn extent(&self) -> f64;
    pub fn streams(&self) -> &[RowPrefix];
    pub fn record(&self, calendar: &CalendarAxis) -> TrainingPartitionRecord;
}
pub enum WholeFiles { NoHoldout, SeparateHoldoutFiles }  // picks the rule `whole` records
pub enum PartitionError { NoTrainingRows { stream }, NothingHeldOut, NoRows,
                          NotAPrefix { stream, position }, Digest(String) }
```

Types are in `rust/crates/cli/src/fit/training_partition.rs` (crate-private).
`whole` returns a `Result` because it digests the rows, and a stream set with no
rows has no extent. There is no `trained(stream, position)`: with prefixes,
membership is `position < rows(stream)`, and the partition is the only thing
that answers it.

- **Removes:** a fit recording a training window it did not use; the two holdout
  forms recording different things; later commands re-deriving membership by a
  different convention.
- **Why a count and a digest:** the digest proves the fitted rows are unchanged
  when a later step reads the file again (rows appended after them do not
  matter; an edited fitted row is refused as changed data). The count makes
  membership "the first _n_ rows", so no consumer compares a time. The digest is
  the score-artifact proposal's row digest, decided as D27.
- **Invariant:** for each stream the training rows are a prefix. Ends increase
  strictly, so the rule in §4.1 always yields a prefix; the constructor checks
  it rather than assuming it.
- **Built by:** `fit run` and `profile`, through the two constructors only
  (`runner::cutoff_partition` and `runner::whole_partition` wrap them and word
  the refusals). **Read by:** `camdl predict`, in every mode, through
  `TrainingPartition::read`, which refuses data whose leading rows no longer
  digest to the recorded prefix; a predicted row's `role` is its position
  against its stream's prefix (`RowRole::of_data_row`). The forecast plan takes
  its parameter set and its origin from the partition read this way.
  `pfilter --save-final-state` cuts its streams by the same rule
  (`training_partition::cutoff_prefix`).
- **Stored:** in `fit.meta.json`, for every fit (`TrainingPartitionRecord`,
  `run_meta.rs`), and copied as `parameter_set` into `predictive.json` and
  `forecast.json`. `rule_version` is `PARTITION_RULE_VERSION`, the version of
  the rule that decides membership (§5.4); a `date` is present when the model
  has an `origin`:

```json
"training_partition": {
  "rule_version": 1,
  "rule": {"kind": "cutoff", "at": {"t": 24.0, "date": "2026-07-25"}},
  "extent": {"t": 24.0, "date": "2026-07-25"},
  "streams": [{"stream": "cases", "rows": 3, "digest": "9f2c…"}]
}
```

### 3.3 `ForecastPlan`: what is to be predicted

Types are in `rust/crates/cli/src/fit/forecast_plan.rs` (crate-private).
`StateConditioning` holds `RowPrefix` (§3.2); it is built by the forecast driver
(`fit/forecast_driver.rs`) and returned with the forecast, beside the plan. Only
a fixed-origin forecast on the chain-binomial backend builds a plan; the plan in
the code has no `mode`, `notes` or `required_times`, and `StateConditioning` has
only its `Filtered` variant.

```rust
pub struct ForecastPlan {
    mode: PredictiveMode,              // InSample | OneStepAhead | FixedOrigin
    origin: f64,                       // forecast origin
    horizon: Option<f64>,              // reach as a duration; None outside FixedOrigin
    parameters: TrainingPartition,     // from the fit record, digest-verified
    state: StateConditioning,          // from what the filter or sampler reports
    targets: Vec<Target>,              // sorted by (end, stream)
    notes: Vec<PlanNote>,              // future covariates, stale streams
}
pub enum StateConditioning {
    None,                                          // ODE, or free-running from the start
    Filtered { through: f64, rows: Vec<RowPrefix> }, // rows assimilated per stream
}
pub struct Target {
    pub stream: String,
    pub measures: Measures,
    pub horizon: i64,                  // position among the stream's targets; 0 = first ending after the origin
    pub lead: f64,                     // end − origin
    pub observed: Option<usize>,       // position of a held-out row, when one exists
}
impl ForecastPlan {
    pub fn new(...) -> Result<Self, PlanError>;   // checks the invariants below
    pub fn required_times(&self) -> Vec<f64>;     // starts, ends, origin: snapshot times
}
```

- **Removes:** different target rules per command; targets wider than the
  observed rows; a period dropped silently after a mid-period origin; a band
  that filtered through held-out rows presented as a forecast.
- **The seam:** every predictive command builds one plan and hands the engine
  `required_times()` and, per target, its start and end. Flow over a target is
  the difference of the stream's cumulative projection at those two times. No
  engine's inner loop changes.
- **Targets are declared before filtering:** `ForecastPlan::new` turns every
  target into a declared period of its stream before the filter runs, and the
  forecast driver reads a stream's running total only at the end of a period the
  plan declared. Callers do not declare periods. The reason is measured: a
  stream's running total is reset only where a declared period starts
  (`MultiStreamObsModel::reset_due_acc`), so if the week under way is not
  declared, the total at the origin covers the previous week as well and the
  prediction is wrong without any error (§4.4).
- **Who may build the conditioning fields:** `parameters` is read only from a
  fit record whose prefix digests verify against the data read. `state` is built
  only from what the filter or path sampler returns; no caller passes it in. A
  label therefore cannot claim conditioning that did not happen.
- **Checked by `new`:** each target ends after the origin; targets of one stream
  do not overlap and are strictly increasing; under `FixedOrigin` each target
  ends at or before origin plus horizon; the origin is not earlier than the end
  of the last row in the state set; the state set's prefixes agree with the
  parameter set on the rows they share.
- **Stored:** in every predictive and forecast-score artifact.

### 3.4 `CalendarAxis`: the mapping from model time to dates

```rust
pub struct CalendarAxis {            // in ir::caltime
    pub time_unit: TimeUnit,
    pub origin: Option<IsoDate>,     // None: the model has no origin
}
impl CalendarAxis {
    pub fn from_model(m: &Model) -> Result<Self, CalError>;
    pub fn parse(time_unit: &str, origin: Option<&str>) -> Result<Self, CalError>;
    pub fn days_per_unit(&self) -> f64;
    pub fn date_of(&self, t: f64) -> Option<RenderedInstant>;      // None: no origin, or t not finite
    pub fn parse_instant(&self, s: &str) -> Result<f64, CalError>;
    pub fn to_json(&self) -> serde_json::Value;
}
```

`date_of` writes `YYYY-MM-DD` for a time on a whole day and `YYYY-MM-DD+<frac>d`
inside a day (`2020-01-01+0.25d`); `parse_instant` reads both forms. The
renderer tries neighbouring doubles so the text reads back to the same `f64`.
Under a `weeks` time unit a double that is not a quotient of any double by 7 has
no exact reading; the renderer then writes the date that reads back closest, one
double away.

- **Removes:** artifacts that cannot be dated without the model source; a
  rendered date that cannot be read back.
- **Why a new type:** `ir::caltime` has the conversions but no struct for the
  axis, so each consumer reassembles it from the IR. One type gives every
  artifact the same block and owns the render/parse pair, so a rendered date
  parses back to the same model time by construction.
- **Stored:** a `calendar` block in the artifacts listed in §5.3, written as
  `{"days_per_unit", "origin", "time_unit"}` (`origin` null for a model without
  one). `days_per_unit` is derived from the unit, written so a consumer needs no
  unit table, and not read back. The IR's `origin_rata_die` is deliberately not
  exported: it is keyed to an internal epoch and would mislead a consumer that
  took it for a standard day number.

### 3.5 What is deliberately not a type

- **No wrapper per time role** (model time, label time, scoring time). The label
  and the period's end are both properties of one row; the confusion is removed
  by not exposing the label as a time, not by wrapping `f64`. Inside the engine
  every time is model time, and the existing `OutputTimes` / `EffectTimes` /
  `ObsTimes` axis wrappers already cover swappable time vectors.
- **No observation identity beyond (stream, position, prefix digest).** Every
  set this design needs is a prefix of a stream's rows (Part 7 says how a
  general identity could be added later).
- **No instant/duration tags in the IR.** The compiler owns that distinction and
  the IR carries resolved numbers.

## Part 4. Rules

### 4.1 Which rows train (D38)

Under `holdout_after`, a row trains if the end of what it measures is at or
before the cutoff. Under `[data.holdout]`, every row of the training files
trains, and every holdout row must end after the training extent, the latest end
among the training rows of every bound stream, or the fit is refused. "At or
before" tolerates float noise of `1e-9` (`TIME_TIE_EPS`), so a date converted to
model time cannot move a row across the cutoff. The partition is decided once,
at fit time, and recorded (§3.2).

An `NA` row is a row: it keeps its position in its stream's training prefix and
is hashed into the row digest. It is not an observation, so under
`holdout_after` a stream whose training rows are all `NA` is refused as having
no training rows, and a cutoff whose held-out rows are all `NA` is refused as
holding out nothing.

For `closing_at` and window-column streams the label is the period's end, so a
rule that compared labels with the cutoff would train the same rows. For `day`,
`starting_on` and `ending_on` streams it would not: the row whose period
straddles the cutoff is held out, although its label is at or before the cutoff.

### 4.2 Which rows a held-out score scores

A score is stamped held-out only when, for every stream, the first `rows` rows
of the data it reads digest to the fit's recorded prefix. It then scores every
row that did not train and whose end is after the training extent. Under
`holdout_after` every untrained row ends after the cutoff, so this is simply
every row that did not train, and the scored set of one stream does not depend
on which other streams the fit binds. Under `[data.holdout]` the fit refuses a
holdout row that ends at or before the training extent (§4.1), so the extent
condition removes no row there either. A `[data.holdout]` fit whose training
file has rows past its recorded prefix is refused, since those rows lie in the
window the score treats as held out. `camdl score` refuses an outcome file whose
leading rows differ, by row digest, from the rows the prediction conditioned on
(`score_command::check_agreement`), and `camdl compare` reads stored scores;
`compare --in-sample` compares the training rows as the in-sample scores they
are. A row whose period overlaps another stream's training days is scored: the
row itself was never fitted, and overlap with other streams' data is the
ordinary condition of every multi-stream score.

### 4.3 Where targets come from

For each stream, one function builds the targets after the forecast origin
(D80):

1. **The declared tiling**, for a stream with a uniform `covers` form: the
   periods continue the stream's tiling at its declared width. Because a uniform
   stream's periods tile without gaps (§2.2), its phase is fixed by any of its
   rows; the first target is the period under way at the origin.
2. **The emit schedule**, for a stream with window columns and an
   `emit_schedule`: the periods between consecutive schedule times. The emit
   schedule already means "the times at which this stream reports" for
   `simulate --obs`; forecasting reads the same declaration. Where the periods
   the stream's rows imply and its schedule disagree over the same reporting
   interval, the forecast and a restart are refused, naming the stream, both
   periods and the source of each (D82).
3. **Held-out rows**, when scoring a window-column stream with no
   `emit_schedule`: the periods of the rows that did not train, validated for
   order, uniqueness, overlap and endpoints. Nothing is extrapolated.
4. **Refusal** otherwise (D42): a window-column stream with no `emit_schedule`
   declares no rhythm, and the forecast tail is refused with a message naming
   the two ways to declare one. A stream with no bound rows is refused too: it
   has no phase.

When scoring a stream with a declared cadence (1 or 2), the held-out rows only
select which targets are scored; they do not renumber them, so a missing week
does not shift the horizon of later weeks. A held-out row whose period is not
one of the stream's targets is refused, never skipped. Every target carries its
lead time (its end minus the origin) apart from its ordinal: for targets taken
from irregular held-out rows (3), the ordinal is not a horizon comparable with a
tiled stream's, and comparisons across data sets with missing periods use lead
time.

The most common gap between observations is never a target source. The horizon
only truncates the list.

Calendar-month and standard-week periods cannot be declared as a uniform
`covers` form (D45). A monthly stream therefore uses window columns, and its
targets come from rule 2 or 3 or it is refused (rule 4); a 30-day step would be
a different target.

### 4.4 Computing a target's value

**Stochastic models (chain-binomial, fitted by PMMH, PGAS or IF2).** For each
posterior parameter draw, or once at the winning point of an IF2 fit, the
forecast driver runs the existing bootstrap particle filter over the observed
rows that end at or before the forecast origin, with the plan's targets appended
to their streams as declared periods that carry no value. Rows ending after the
origin are not given to the filter, so they cannot be assimilated. After the
origin the filter propagates every particle without reweighting: a period with
no value adds nothing to the likelihood. At each target's end it draws the
observable from the stream's observation model (`obs_model.sample`) given that
particle's projection. A forecast is therefore a draw of the observed quantity,
with process noise, parameter uncertainty and the declared observation noise.

A target that began before the origin needs no special handling. Each particle
carries its stream's running total since the period began; it travels with the
particle through resampling, so at the target's end it holds that particle's
inferred flow before the origin plus its simulated flow after it. The part
before the origin is a latent quantity conditioned on the other streams'
observations through the origin, not a known value. The stream's own row for
that period cannot enter the filter before the period ends, and predictions are
drawn before any reweighting at that step.

This was checked by experiment on a chain-binomial SIR with a daily and a weekly
incidence stream, forecast origin day 24, target week days 21–27, fixed
parameters:

| Method                                                                           | Mean of the weekly flow | Variance      |
| -------------------------------------------------------------------------------- | ----------------------- | ------------- |
| Importance sampling from the prior, weighted by the daily data (2,000,000 draws) | 68.227 ± 0.034          | 189.84 ± 0.55 |
| The filter with the target declared as a period                                  | 68.245 ± 0.021          | 190.41 ± 0.36 |
| The same target not declared (no reset at day 21)                                | 256.951 ± 0.043         | 530.87 ± 1.07 |

With matched random streams, stopping at day 24 and continuing was bit-identical
to running through (12 of 12 particle comparisons), so the flow before the
origin is neither reset nor counted twice. The prior mean is 123.1, so the daily
data's conditioning is doing real work. The test
`the_week_under_way_matches_the_importance_sampling_reference`
(`rust/crates/cli/src/fit/forecast_driver_tests.rs`) repeats the comparison
through the forecast driver, for the latent flow and for the reported count with
its observation noise.

**ODE models.** The state is fixed by the parameters; a flow target over
`[a, b)` is the difference of the cumulative flow at `b` and `a` on each
posterior draw's trajectory, both recorded snapshot times, plus a draw from the
observation model. ODE forecasts are unsupported (§2.4).

**How far the run goes.** A forecast runs to the latest end among its targets,
even past the model's `simulate { to }`, and builds its intervention plan to
that end (D49). Extending the run extends nothing else: a recurring schedule
with a literal end keeps it, and a forcing or B-spline evaluated past its range
is the error of §4.6 unless overridden.

**Draws and failures.** Posterior draws are pooled with equal weights: the
forecast approximates ∫ p(y_future | θ, training rows) · p(θ | training rows)
dθ, with the state conditioned on the training rows (D53, D57). No draw is
dropped: a draw whose filter collapses, or any particle that dies after the
origin (a numerical failure, `NumericalCollapse` or a negative count; not
epidemic extinction, which is a valid state), fails the forecast with a
diagnostic naming the draw, the time and the cause.

**Joint samples.** After the origin no stop holds an observed value, so the
filter does not assimilate there and skips the resampling call (D72): no
particle is copied into another's slot, and particle _i_'s draws at successive
targets form one joint sample, with observation noise drawn from that particle's
own trajectory. Joint samples, such as a hub's "sample" output, are therefore
exposed (D54); since any particle death after the origin already fails the
forecast, every exposed sample is a complete trajectory.

**The log score.** Quantiles, the weighted interval score and the CRPS come from
the draws. The log score needs the predictive density at the observed value,
`log p̂(y) = logsumexp_i log p(y | x_i, θ_i) − log N` over equally weighted
particles, evaluated with the same likelihood code as fitting and before the
observation could be assimilated. The filter records each particle's
log-likelihood of a target's held-out value without letting it enter any weight,
and `camdl score` takes the log score from those records
(`fit/forecast_score.rs`), so scoring a forecast does not change it (D56).

**Restarts.** `simulate --init-state` restores compartment counts only, not the
running totals or which period each belongs to. It therefore refuses to emit a
period that began before the restart, with a message, rather than dropping it or
writing a partial total. Forecasts do not go through it.

### 4.5 Alignment with a step grid (D43)

Whether a period's start and end may lie between steps is a capability of the
method, recorded once per method in `fit::methods::METHODS` as
`PeriodBoundaries`:

- `Anywhere` (if2, pfilter, pmmh): steps exactly to every start and end of every
  observed period, shortening the step where a boundary falls between grid times
  (`sim::schedule::Schedule::exact_inference`).
- `OnGrid` (PGAS): requires every start and end to lie on the grid
  `t_start + k·dt`, within `sim::schedule::GRID_TOL` steps, and otherwise
  refuses the run before it starts, naming the stream, the row, the boundary and
  the two grid times either side (`runner::refuse_periods_off_the_grid`).
- `AtOutputTimes` (the ODE methods: nl-sbplx, nl-bobyqa, nl-lbfgs, mh): reads
  the integrated trajectory at its output snapshots; a boundary with no snapshot
  is refused by the trajectory walk (`sim::inference::ode_loglik`), naming the
  time.

A period is never moved to a different period to fit a grid. This is a
capability of the method, not a user setting. The check runs once the method is
known, measured from the run's start.

### 4.6 Time functions outside their range, and future covariates (D44, D48)

A time function evaluated outside the range on which it is defined is an error,
unless the run is given the override `--allow-forcing-extrapolation` (in a fit
config, `[config] allow_forcing_extrapolation = true`). The commands that run a
model take it (`args::ForcingExtrapolationArg`). This covers forcings built from
a data table (interpolated, cubic, piecewise) before their first or after their
last data point, and B-spline forcings outside their declared window. Under the
override the function holds its end value and the run says so once per forcing
and side, as a warning with notice code `forcing-outside-range` naming the
function, its range, the side evaluated and the span held. Formulas defined for
all time and periodic forms are unaffected.

The override is identity: a `simulate`, `pfilter` or `profile` run under it keys
differently from the same run without it (`SimConfig` and the `pfilter` and
`profile` configs carry `allow_forcing_extrapolation`; a fit keys it through its
config blob), and every `run.json` such a command writes records it as
`forcing_extrapolation: {allowed: true, held: [...]}`, each held entry giving
the forcing, the range it is defined on, the side and the value held. A run
without the override records nothing: it held nothing, or it was refused before
it committed. `score.json` records, as `forcings_held`, each forcing held while
the prediction was computed again.

The check reads the values the run actually evaluated, not the run's time span:
a forcing referenced only under a condition that is never true past its range
does not trigger it. The engine owns the record, not the policy.
`sim::forcing_range::DefinedRange` holds one set-once flag per side of each
compiled forcing, shared by every clone of the compiled model; a clamped
evaluation loads the flag and writes it only while it is clear, so it is written
once per process and particle threads never contend. A time within the rounding
tolerance of an end counts as the end, not past it. The first evaluation on a
side emits a `RunNote::ForcingOutsideRange` (`sim::run_notes`), and the
command's reporter (`cli::forcing_extrapolation::report`) turns it into the
error or the warning. Because the flag records a side and not a time, the
message names the side; which time outside the range a multi-threaded run meets
first depends on its threads. A command that holds a store claim while its model
runs (`fit run`) defers the error so the claim is failed through its own path.
For a forcing with a `lag`, the range bounds its argument, `t − lag`, and the
message says so.

Separately, a forecast that reads a data-table forcing's values after the
forecast origin uses covariate values that would not have been known at the
origin, if they are observations rather than a scenario. camdl cannot tell
which, so the forecast is refused before any filter runs until each such forcing
has a statement, `--covariate FORCING=prospective|realised|hold`, judged on the
table's argument `t − lag` (`cli::fit::forecast_covariates`). The statements key
the prediction, and `forecast.json` records them; `realised` labels the forecast
conditional on covariates observed after the origin, and `hold` holds the
compiled forcing at its value at the origin (`sim::compiled_model::HeldAfter`).
The rules are the run spec's, §2.2.2, "Forcings read after the origin".

### 4.7 `last_obs` (D40)

`last_obs` is the latest end of an observed period, and `first_obs` the earliest
start, one value per fit across all bound streams. An `NA` row counts toward
neither. For `holdout_after = "last_obs - …"` (and `first_obs`) they are
computed over the full data files, because they define the split
(`runner::resolve_holdout_cutoff`); for every other anchor
(`simulate { to
= last_obs + … }`, breakpoints, `value_at`) over the training
rows only, so appending held-out rows never changes a fit's end time or its
identity. A fit prints a note, once, naming any stream whose last observation is
more than one of its own periods before `last_obs` (`runner::partition_lines`),
so a stream that stopped reporting is visible. A forecast prints no such note.
`camdl predict --mode free_forward` builds no forecast plan: past a stream's
data, its grid continues the most common gap between the stream's observation
times (`fit::predict::forecast_times`).

## Part 5. Boundaries, artifacts and identity

### 5.1 What each layer owns

- **The compiler** owns the source-language rules for time and checks them
  statically: a calendar duration never shifts an instant except through the
  named functions, in every position including a `covers` width;
  `Instant +
  Instant`, `k · Instant` and `Duration − Instant` are refused;
  `date_range(…,
  calendar_months = n)` computes each date from its start. It
  lowers every temporal expression to model time.
- **The IR** carries what the runtime needs: the calendar axis (`time_unit`,
  `origin`, `origin_rata_die`), each stream's `covers`, column roles, emit
  schedules and forcing knot times. It has no calendar `covers` form (D45).
- **The runtime** owns everything about data, fits and forecasts. Each runtime
  type is built at one boundary, and the engines receive `f64` vectors. The CLI
  time grammar (`runner::parse_time_spec`) is the runtime's one parser for times
  a user types; `--horizon "4 weeks"` is read by `forecast_plan::parse_reach`,
  with the same unit words (`runner::canonical_duration_unit`). Without an
  `origin` their months and years are affine spans, as the IR's are; under an
  `origin` a calendar-month or year horizon is refused.

### 5.2 Invariants and who enforces them

"Static" means a violating value cannot be constructed; "checked" means a
constructor refuses it.

| Invariant                                                                              | Owner                                              | Kind                                                                      |
| -------------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------- |
| A calendar duration shifts an instant only through `add_calendar_*`, in every position | `ocaml/lib/compiler/time_typing.ml`                | static                                                                    |
| Instant arithmetic follows `docs/dates.md`, "Three concepts"                           | `time_typing.ml`                                   | static                                                                    |
| `date_range` calendar stepping is computed from its start                              | `ocaml/lib/compiler/expander.ml`                   | static                                                                    |
| A period has finite, increasing endpoints                                              | `Period::new`                                      | checked                                                                   |
| A row's period is computed in one place                                                | `observed_rows`                                    | static (only producer)                                                    |
| Training is decided on periods, never labels                                           | `TrainingPartition::from_cutoff`                   | static (no label input)                                                   |
| Training rows are a prefix of each stream                                              | `TrainingPartition`                                | checked                                                                   |
| A prediction's fitted rows digest to the fit's prefix                                  | `TrainingPartition::from_record`                   | checked                                                                   |
| A score's outcome file begins with the rows the prediction conditioned on              | `camdl score` (`check_agreement`)                  | checked                                                                   |
| Targets end after the origin, do not overlap, and stay within the horizon              | `ForecastPlan::new`                                | checked                                                                   |
| Every running total read for a prediction belongs to a period the plan declared        | `ForecastPlan::new` and the forecast driver        | static (the driver declares the periods before filtering; callers cannot) |
| No observed row ending after the origin reaches the filter                             | forecast driver                                    | checked (a likelihood increment after the origin fails a test)            |
| A restart never emits a period that began before it                                    | `obs_emit::refuse_period_under_way`                | checked (refusal)                                                         |
| Conditioning fields come from the fit record and the filter, not a caller              | `ForecastPlan::new`, `forecast_driver`             | static                                                                    |
| A period is never moved to fit a step grid                                             | alignment check                                    | checked (refusal)                                                         |
| A time function outside its range is an error unless overridden                        | `sim::forcing_range`, `cli::forcing_extrapolation` | checked                                                                   |
| A rendered date parses back to the same model time                                     | `CalendarAxis`                                     | checked (property test, `ir/tests/calendar_axis_roundtrip.rs`)            |
| Every artifact carrying a model time carries the calendar axis                         | `RecordMeta::new`, `FitSidecar.calendar`           | static: a required constructor argument and field                         |

### 5.3 Artifact contracts

Every artifact that carries a model time carries the `CalendarAxis` once, and
names each instant as `{t, date}`, with `date` recorded or explicitly "not
recorded" for a model without an origin.

- **Run record** (`run.json`, every kind): carries `calendar`.
- **Fit record** (`fit.meta.json`): carries `training_partition` and `calendar`.
- **Predictive artifacts** (`camdl predict`, `simulate --init-state`): the plan
  (mode, forecast origin, horizon, both information sets, notes) in the JSON;
  per row in the TSV, a `role` (`trained`, `held_out`, `target`), the period
  (`t_start`, `t_stop`, and dates when the model has an origin), and for targets
  `horizon` and `lead`.
- **Score artifact** (`camdl score`; `2026-11-23-prediction-and-scores.md`):
  each scored target records the forecast origin, its horizon and its lead;
  scores are kept per forecast date and horizon and never pooled across horizons
  (D35); the fit's conditioning travels with the score.

### 5.4 Identity

The training rule is part of a fit's identity: a change to which rows train
changes the rows a fit is fitted to, so it re-keys the fit. The forcing-
extrapolation override is part of the identity of `simulate`, `pfilter` and
`profile` runs (§4.6).

A fit problem's identity holds each data file's training lines and each stream's
held-out dates without their values (`fit_store::FitProblemIdentity`), so two
partitions of one file pose two problems and a held-out value is in no identity.
The partition record carries the rule's version (`PARTITION_RULE_VERSION`) and
each stream's training row digest. A forecast's mode, horizon and target list
are part of the prediction's identity (`fit::predict::PredictOptions`, whose
`ForecastKey` holds `ForecastPlan::targets_digest`, the design mode and the
covariate statements); its origin is fixed by the fit it reads; notes are
presentation.

### 5.5 Names

| Concept                                                                     | Name                                         | Not to be called                                          |
| --------------------------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------- |
| the calendar date of `t = 0`                                                | origin                                       | —                                                         |
| the instant after which a forecast sees no data                             | forecast origin                              | "origin" alone                                            |
| the date or number printed in a row's time column                           | label                                        | "time" of the row                                         |
| what a row measures                                                         | its instant or period; the code's `Measures` | —                                                         |
| the latest end among training rows                                          | training extent                              | `train_end` as a label                                    |
| the instant `holdout_after` names                                           | cutoff                                       | `score_from` as a user concept                            |
| a target's count of its stream's periods from the one containing the origin | horizon                                      | "step"; a hub's horizon, which export computes separately |
| how far ahead a forecast reaches, as a duration (`--horizon "4 weeks"`)     | horizon (on the command line)                | —                                                         |
| the duration from the forecast origin to a target's end                     | lead time                                    | —                                                         |
| the predictive mode flag of `camdl predict`                                 | `--mode`                                     | `--horizon`                                               |
| the count of scored rows in `compare`                                       | scored steps                                 | horizon                                                   |

## Part 6. Worked examples

**A day-labelled stream across a cutoff.** `covers = day(time)`, labels 1–16,
`holdout_after = 10`. Row 9 counts day 9 and ends at 10, so it trains; row 10
counts day 10 and ends at 11, so it is held out. The training extent is 10. The
fit reports "trained on 9 rows, through day 10".

**Two cadences.** A weekly `closing_at(time, 7 'days)` stream labelled 7–70 and
a daily `day(time)` stream labelled 0–69, `holdout_after = 24`. Daily rows 0–23
train (ends 1–24); weekly rows 7, 14 and 21 train (ends at or before 24). The
extent is 24. Weekly row 28, covering days 21–27, did not train and is scored,
like every daily row from 24 on.

**A monthly series.** Window-column rows ending at days 31, 59, 90, 120, 151 and
181, no `emit_schedule`, no held-out rows. A forecast is refused: "stream
'cases' is refused a forecast: its rows state their own windows (window_start /
window_stop), so there is no rule that says which period comes next", followed
by the two ways to declare one.

**A period moved by a grid (invalid).** PGAS, `simulate { from = 0.5 'days }`,
`dt = 1 'days`, a `closing_at(time, 2 'days)` stream with labels 3, 5, 7. The
first period, days 1–3, does not lie on the grid 0.5, 1.5, 2.5, …. Refused,
naming row 1 (time = 3), its start, 1, and the grid times 0.5 and 1.5. The same
data under if2 or pmmh runs, stepping to the boundaries.

**A forcing past its data.** An interpolated temperature forcing with data
through day 100, a run to day 130: an error naming the forcing, its range (up to
day 100) and that the run evaluated it after its last data point, unless
`--allow-forcing-extrapolation` is given, in which case the run warns, naming
the span held (from day 100 to the end of the run), and records it in
`run.json`. A forecast from origin day 60 that reads the forcing over days
60–100 is refused until the forcing has a `--covariate` statement (§4.6).

## Part 7. Deferred, and why this design does not block it

| Later work                                                            | What it adds                                                                                                          | Why nothing here is rewritten                                                                                                            |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Calendar `covers` forms (D45)                                         | `month(time)`, `iso_week(time)`, `mmwr_week(time)`; a load-time check that labels fall on the right day               | a calendar width is one more declared tiling (§4.3, case 2); one IR variant                                                              |
| Aggregated targets                                                    | weekly targets from a daily stream                                                                                    | a target is a stream and a period; an aggregate is a new target kind beside it                                                           |
| Observation identity across datasets                                  | a key per row independent of position                                                                                 | consumers read membership only through `RowPrefix` and `TrainingPartition::rows`; a key set can replace the count behind the same method |
| Data vintages and `as_of` reconstruction                              | "the data as known at the forecast origin"                                                                            | the two information sets are already separate; a vintage becomes one more field of `RowPrefix`                                           |
| Hubverse import and export                                            | external forecasts scored on camdl targets                                                                            | a target's stream, end date and horizon are a hub's join key; the score artifact stores the hubs' 23 quantile levels                     |
| A per-forcing extrapolation declaration                               | `extrapolate = hold` in the model                                                                                     | replaces the run-level override for that forcing; the error remains the default                                                          |
| Conditioning the state on rows after the fit (rolling forecast dates) | D35's per-date state conditioning without a refit                                                                     | the plan already records the parameter and state sets separately; the driver filters further rows with fixed parameter draws             |
| Fixed-origin forecasts of ODE fits (MH, NLopt)                        | the difference of the cumulative flow at a target's two ends on each draw's trajectory, plus observation noise (§4.4) | a target is a stream and a period whatever computes its value; the state set is recorded as empty                                        |
| Filtering once and forking scenarios at the origin                    | scenario arms without repeating the historical filtering                                                              | running totals live in each particle in memory, so the swarm can be copied per arm                                                       |
| Standalone external-forecast scoring                                  | scoring hub files against hub target data with no camdl fit                                                           | a target is a stream, an interval and an observed value; a target-data reader supplies them instead of a fit                             |
| Restarts that continue a period under way                             | `simulate --init-state` carrying running totals                                                                       | a new saved-state version with each total's stream and period start; forecasts never depend on it                                        |
