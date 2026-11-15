# Temporal architecture: observation periods, training data and forecasts

**Status:** Normative design, decided 2026-11-15 (decisions D35 and D38–D48 in
the maintainer's docket). Not yet implemented; the staged plan that implements
it is `docs/dev/proposals/2026-11-15-temporal-architecture-migration.md` (the
migration plan). Until a stage lands, the code behaves as described in Part 1.

**Reader.** A contributor who needs to know how camdl represents time from a
data file through a fit to a forecast and its score, and which code owns each
rule. The modeler's account of calendar time is `docs/dates.md`; this document
assumes its vocabulary (instant, duration, period, `origin`, `time_unit`,
`covers`) and does not repeat it.

## Summary

Every decision about a data row is made from the stretch of time the row
measures, never from the date printed in its time column. Training membership is
decided once, recorded as "the first _n_ rows of each stream" with a digest of
those rows, and read by every later step rather than re-derived. Every
prediction is made against an explicit list of targets, each a stream and a
period, built once in a forecast plan that every backend computes against. The
simulation engines keep plain floating-point times; the new structure sits
between the IR and the engines.

## Part 1. The existing design and its problems

### 1.1 What stays: the calendar is erased at the compiler and loader

The compiler checks the source-language rules for time (an instant is not a
duration; a calendar month cannot shift an instant except through
`add_calendar_months`; `date("2026-07-05")` is the instant 5 July begins) and
lowers every temporal expression to model time, a number in the model's
`time_unit` counted from `origin`. The data loader converts dates in data files
the same way. Below those two places every time is an `f64` (`docs/dates.md`,
"The one rule"). This erasure is correct and this design keeps it: the engines
have no use for calendar structure, and a second time language in Rust would be
a second definition of one semantics.

The runtime already has the right primitive for a data row's time span: `Period`
(`rust/crates/sim/src/inference/multi_stream_obs.rs:1434`), a half-open interval
whose only constructor refuses a non-finite or non-increasing pair, and
`StreamTimes`, which holds a stream's instants or periods. `Covers::period_of`
(`rust/crates/ir/src/observation.rs:304`) turns a label into a period under each
`covers` form.

### 1.2 Where meaning is lost

Measured at `main` `0ed74876d`; the migration plan's Part 1 has the evidence and
reproductions.

- **The label is compared with the cutoff.** The loader keeps only each row's
  label as its time (`Observation.time`), and the training cut compares that
  label with `holdout_after` at six separate sites. Under a `covers` form whose
  label is the start of the period, a fit trains on data past its cutoff and
  records a training window it did not use.
- **Forecast targets come from three rules in four places.** `fit predict`
  continues the most common gap between observations; `simulate --obs` and
  `--init-state` use the declared `emit_schedule`; `--design-from` reproduces
  the fit's rows. Monthly series drift (gh#861).
- **What conditioned a prediction is not recorded.**
  `fit predict --horizon
  one_step` filters through held-out rows and nothing
  marks them.
- **Alignment is checked before the method is known.** Off-grid observations are
  refused even for methods that step exactly; PGAS rounds an observation's
  period onto its grid without notice.
- **Forcings built from data hold their last value silently** past their last
  data point.
- **Artifacts cannot all be dated.** The stored simulation record carries no
  calendar; sub-day dated output cannot be read back.

![Figure 1: four rows under a cutoff at day 24. The same week, days 21 to 27, is row 21 under starting_on and row 28 under closing_at; under day(time), row 23 trains and row 24 is held out. Today the starting_on row and the day-24 row train because their labels are at or before 24.](figures/temporal-row-periods.svg)

_Figure 1. A row's period, not its label, decides training._ All four rows are
judged against `holdout_after = 24`. Today two of them train because their label
is at or before 24, although their periods end after it. The rule in §4.1 holds
both out.

## Part 2. Concepts

### 2.1 The time axis

Model time `t`, `origin` and `time_unit` are as in `docs/dates.md`. `origin`
only says which calendar date is `t = 0`; changing it renumbers every time and
moves nothing in calendar time. It is unrelated to the forecast origin (§2.5),
and the word "origin" alone always means this one.

### 2.2 What a data row measures

A row has a **label**, the number or date in its time column, and measures
either an **instant** (a state read, such as a census: `projected = I`) or a
**period** (a count accumulated over a stretch of time: `incidence(...)`). For a
period, the stream's `covers` line says how the label maps to the period:

| `covers` form                  | label 21 means        | the period's end |
| ------------------------------ | --------------------- | ---------------- |
| `day(time)`                    | day 21                | 22               |
| `starting_on(time, 7 'days)`   | days 21–27            | 28               |
| `ending_on(time, 7 'days)`     | days 15–21            | 22               |
| `closing_at(time, 7 'days)`    | days 14–20            | 21               |
| `window_start` / `window_stop` | read from two columns | `window_stop`    |

The **end** of what a row measures is the instant for an instant row and the
period's end for a period row. It is when the row's information is complete, and
it is the time at which the row is scored (`StreamTimes::closes`,
`multi_stream_obs.rs:1520`). Every temporal decision in this design reads the
period or its end; the label survives only as provenance, inside the row digest.

A row's **position** is its index among its stream's rows in file order. The
binder requires each stream's ends to be strictly increasing
(`multi_stream_obs.rs:1003`), so position order is time order. A uniform
`covers` form also forbids gaps (`rust/crates/cli/src/pfilter.rs:1606`: a gap is
a missing row; an unobserved period is written as an `NA` row), so a uniform
stream's periods tile without gaps or overlaps.

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

A prediction depends on two different sets of observations, recorded separately
because they legitimately differ:

- the **parameter set**: the rows the parameters were fitted to (the training
  rows);
- the **state set**: the rows the latent state at the forecast origin was
  conditioned on, by a filter or by sampled paths.

Under D35 the parameters are fitted once, and the state may then be conditioned
on newer rows up to each forecast date without a refit, so the state set can
extend past the parameter set. For a free-running projection from the start of
the run, the state set is empty.

What the state set means depends on the backend:

- **Chain-binomial (particle filter):** given the parameters the state is still
  random. A filter run through the rows up to the forecast origin leaves
  particles that approximate the state's distribution given those rows, and the
  forecast continues each particle.
- **PGAS:** each posterior draw carries a sampled path, so a forecast pairs each
  parameter draw with its path's state at the origin, carrying parameter and
  state uncertainty together.
- **ODE:** given the parameters the state is fully determined, so conditioning
  on observations after fitting changes nothing; the data reach an ODE forecast
  only through the parameters. The state set is empty for every ODE forecast.
  Consequently, rolling ODE forecasts with fixed parameters are slices of one
  trajectory and do not use newer data, while a stochastic model at the same
  forecast dates does. Comparing the two is fair, since the difference is the
  model's, and the artifact says which kind each forecast is.
- **Gillespie:** no forecast path exists; a forecast on this backend is refused.

### 2.5 Forecast origin, targets, horizon and lead time

The **forecast origin** is the instant after which a forecast sees no data. By
default it is the latest end among the rows of the state set. A user may set a
later one, never an earlier one: an origin earlier than data the state has seen
is refused.

A **target** is one stream's period (or instant) that the forecast predicts. A
target exists whether or not an observation for it exists; when one does, the
target carries the observed row's position.

A target's **horizon** is counted the way forecast hubs count it: as a number of
the stream's own periods, starting from the period under way at the forecast
origin, which is horizon 0. In the CDC FluSight and COVID-19 hubs,
`target_end_date = reference_date + horizon × 7 days`, horizon 0 is the week
containing the reference date and −1 the week before; the unit is set per
target, and the Reich-lab COVID hub counted "N wk ahead" and "N day ahead"
targets side by side, each in its own unit. In camdl a stream's unit is the
width its `covers` line declares, so a weekly stream counts weeks and a daily
stream counts days, with no separate setting. A target's **lead time** is the
duration from the forecast origin to the end of its period.

On the command line, `--horizon "4 weeks"` is a duration: the forecast includes
every target whose period ends within 4 weeks of the forecast origin. With an
origin on a week boundary, a weekly stream gets horizons 0–3 and a daily stream
horizons 0–27. The two uses of the word (the count on a target, the reach on the
command line) follow the literature, which uses "horizon" for both (Hyndman and
Athanasopoulos, _Forecasting: Principles and Practice_, 3rd ed.).

![Figure 2: a weekly stream; the parameters were fitted to the first four rows, the state filtered through six; the forecast origin is the end of the sixth period; the next four periods are targets with horizons 0 to 3, each with a lead time from the origin to its end.](figures/temporal-information-sets.svg)

_Figure 2. Two information sets and one forecast._ The parameters were fitted
once; the state was then filtered through two newer rows without a refit; the
forecast origin is the end of the last filtered row.

When streams report at different rates, a slower stream's current period has
usually begun before the forecast origin. It is still that stream's horizon 0,
and it is predicted on every backend (D39): the part of the period before the
origin is inferred from the state, and the part after it is predicted.

![Figure 3: a daily stream observed through day 23 and a weekly stream whose last row covered days 14 to 20; the forecast origin is the start of day 24; the daily targets are days 24, 25, 26 with horizons 0, 1, 2; the weekly week of days 21 to 27 is horizon 0, with days 21 to 23 inferred and days 24 to 27 predicted.](figures/temporal-two-cadences.svg)

_Figure 3. Horizons on two streams that report at different rates._ Each stream
counts horizons in its own period; the weekly stream's week under way is its
horizon 0.

### 2.6 Three independent axes of a prediction

| Axis                                        | Values                                                                                                                                                                                                                               | Recorded as                                                       |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| **Mode**: what the prediction conditions on | in-sample (parameters fitted to the predicted rows); one step ahead (parameters exclude the predicted row; state filtered through every earlier row); fixed origin (state conditioned up to the forecast origin, free-running after) | the plan's `mode`                                                 |
| **Parameter uncertainty**                   | one plug-in point, or a mixture over posterior draws                                                                                                                                                                                 | the trace's `Provenance` (exists)                                 |
| **Numerical method**                        | particle filter, ODE skeleton plus observation noise                                                                                                                                                                                 | the trace's `basis` (exists, `rust/crates/score/src/trace.rs:61`) |

A free-running projection from the start of the run is a fixed-origin forecast
whose origin is the run's start and whose state set is empty, not a fourth mode.
No value on one axis implies a value on another.

## Part 3. Type design

![Figure 4: the compiler and IR are unchanged by this design; in the runtime, one row reader produces rows with their periods, the training partition is built from them and written to the fit record, the forecast plan reads the partition and the filter's record of what it assimilated, and the engines receive plain numbers.](figures/temporal-layers.svg)

_Figure 4. Layers, seams and ownership._ The new types form a thin band between
the IR and the engines. Each is built at one boundary; the engines see only
`f64` vectors through the existing `Schedule` and `StreamTimes::closes` seams.

Four runtime types are added or extended. For each: what it represents, the
invalid state it removes, who builds it, who reads it, and how it is stored.
Field names are illustrative; the meanings are normative.

### 3.1 `ObservedRow`: one data row, with what it measures

```rust
pub struct ObservedRow {
    pub position: usize,             // index among this stream's rows (§2.2)
    pub measures: Measures,          // the one temporal fact later code reads
    pub temporal: Vec<TemporalCell>, // the time cells as written; hashed, never compared
    pub cell: Option<ObsCell>,       // None is an NA row
    pub aux: Vec<(String, f64)>,     // e.g. a binomial denominator
}

pub enum Measures { Instant(f64), Period(Period) }  // Period is the existing type
impl Measures { pub fn start(&self) -> f64; pub fn end(&self) -> f64; }
```

- **Removes:** any comparison of a label with a cutoff, because no field holds
  the label as a time.
- **Why the existing types do not suffice:** `StreamTimes` holds periods per
  stream but is built after the training cut, from rows whose only time is the
  label.
- **Built by:** `observed_rows(...)`, the one reader, once per stream; it is the
  score-artifact proposal's `observed_rows` with `measures` in place of its
  `time`.
- **Read by:** the likelihood loader, the training partition, the row digest,
  the forecast plan.
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
    pub fn whole(rows: &BTreeMap<String, Vec<ObservedRow>>) -> Self;
    pub fn trained(&self, stream: &str, position: usize) -> bool;
}
```

- **Removes:** a fit recording a training window it did not use; the two holdout
  forms recording different things; later commands re-deriving membership by a
  different convention.
- **Why the existing type does not suffice:** `TrainingWindow { train_end }`
  (`rust/crates/cli/src/run_meta.rs:626`) is a bare label time with no row count
  or digest, and is absent when no holdout is declared.
- **Why a count and a digest:** the digest proves the fitted rows are unchanged
  when a later step reads the file again (rows appended after them do not
  matter; an edited fitted row is refused as changed data). The count makes
  membership "the first _n_ rows", so no consumer compares a time. The digest is
  the score-artifact proposal's row digest, decided as D27.
- **Invariant:** for each stream the training rows are a prefix. Ends increase
  strictly, so the rule in §4.1 always yields a prefix; the constructor checks
  it rather than assuming it.
- **Built by:** `fit run`, `profile`, `survey`, through the two constructors
  only. **Read by:** `compare`, `camdl score`, `fit predict`, the forecast plan.
- **Stored:** in `fit.meta.json`, for every fit, replacing `training_window`:

```json
"training_partition": {
  "rule": {"kind": "cutoff", "at": {"t": 24.0, "date": "2026-07-25"}},
  "extent": {"t": 24.0, "date": "2026-07-25"},
  "streams": [{"stream": "cases", "rows": 3, "digest": "9f2c…"}]
}
```

### 3.3 `ForecastPlan`: what is to be predicted

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
    pub horizon: Option<i64>,          // count of the stream's periods; None if it has no step
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
  the difference of the stream's cumulative projection at those two times, which
  is how `fit predict` already computes bands
  (`rust/crates/cli/src/fit/predict.rs:1990-2105`). No engine's inner loop
  changes.
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
    pub fn date_of(&self, t: f64) -> Option<RenderedInstant>;
    pub fn parse_instant(&self, s: &str) -> Result<f64, ParseError>;
}
```

- **Removes:** artifacts that cannot be dated without the model source; a
  rendered date that cannot be read back.
- **Why a new type:** `ir::caltime` has the conversions but no struct for the
  axis, so each consumer reassembles it from the IR. One type gives every
  artifact the same block and owns the render/parse pair, so a rendered date
  parses back to the same model time by construction.
- **Stored:** a `calendar` block in every artifact that carries a model time.

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
trains, and every holdout row must end after the training extent or the fit is
refused. The partition is decided once, at fit time, and recorded (§3.2).

For `closing_at` and window-column streams the label is the period's end, so
this rule trains exactly the rows today's label rule trains, and their fits are
unchanged. For `day`, `starting_on` and `ending_on` streams the row whose period
straddles the cutoff stops training.

### 4.2 Which rows a held-out score scores

A score is stamped held-out only when, for every stream, the first `rows` rows
of the data it reads digest to the fit's recorded prefix. It then scores every
row that did not train and whose end is after the training extent. Under
`holdout_after` every untrained row ends after the cutoff, so this is simply
every row that did not train, and the scored set of one stream does not depend
on which other streams the fit binds. Under `[data.holdout]` with training files
that end at different times, the extent also excludes a held-out row that ends
before the latest training data, which the parameters would otherwise have seen
the future of. A row whose period overlaps another stream's training days is
scored: the row itself was never fitted, and overlap with other streams' data is
the ordinary condition of every multi-stream score.

### 4.3 Where targets come from

For each stream, the targets after the forecast origin are, in order:

1. **Held-out rows**, when scoring: the periods of the rows that did not train.
   Nothing is extrapolated.
2. **The declared tiling**, for a stream with a uniform `covers` form: the
   periods continue the stream's tiling at its declared width. Because a uniform
   stream's periods tile without gaps (§2.2), its phase is fixed by any of its
   rows; the first target is the period under way at the origin.
3. **The emit schedule**, for a stream with window columns and an
   `emit_schedule`: the periods between consecutive schedule times. The emit
   schedule already means "the times at which this stream reports" for
   `simulate --obs`; forecasting reads the same declaration.
4. **Refusal** otherwise (D42): a window-column stream with no `emit_schedule`
   declares no rhythm, and the forecast tail is refused with a message naming
   the two ways to declare one. A stream with no bound rows is refused too: it
   has no phase.

The most common gap between observations is never a target source. The horizon
only truncates the list.

Calendar-month and standard-week periods cannot be declared until the calendar
`covers` forms of the follow-up proposal land (D45). Until then a monthly stream
must use window columns and is refused a forecast tail; a 30-day step would be a
different target.

### 4.4 Computing a target's value

A flow target over `[a, b)` is the difference of the stream's cumulative
projection at `b` and at `a`, both recorded snapshot times. This is exact on
every backend at the snapshot times. For a target that began before the origin
(`a` < origin), the part over `[a, origin)` comes from the state: the ODE
trajectory, the PGAS path, or, for the particle filter, a running count since
the period began that the filter's saved state carries (D39). The scientific
target is the same on every backend; the numbers differ by the backends'
legitimate differences.

### 4.5 Alignment with a step grid (D43)

A method that can step exactly (if2, pfilter, pmmh) steps to every start and end
of every observed and target period. A method that cannot (PGAS) requires every
start and end to lie on the grid `t_start + k·dt`, within the named grid
tolerance, and otherwise refuses the run, naming the row and the nearest grid
times. A period is never moved to a different period to fit a grid. This is a
capability of the method, not a user setting; the `[config] obs_alignment` key
is removed. The check runs once the method is known, measured from the run's
start.

### 4.6 Time functions outside their range, and future covariates (D44, D48)

A time function evaluated outside the range on which it is defined is an error,
unless the run is given an override flag (named in the migration plan). This
covers forcings built from a data table (interpolated, cubic, piecewise) past
their first or last data point, and B-spline forcings past their declared
window. Under the override the function holds its end value, the run prints a
warning naming the function, its range and the span held, and the override is
recorded in the run's and the score's artifacts. Formulas defined for all time
and periodic forms are unaffected.

The check reads the values the run actually evaluated, through a set-once flag
in the lookup's clamp branch, not the run's time span: a forcing referenced only
under a condition that is never true past its range does not trigger it. The
flag costs one load per clamped evaluation and is never written twice, so it
adds no contention between particle threads.

Separately, a forecast that reads a data-table forcing's values after the
forecast origin uses covariate values that would not have been known at the
origin, if they are observations rather than a scenario. camdl cannot tell
which. The plan records a note naming the forcing and the span, and the artifact
carries it, so a reader can tell a scenario forecast from one that used realised
future covariates.

### 4.7 `last_obs` (D40)

`last_obs` is the latest end of an observed period, one value per fit across all
bound streams. For `holdout_after = "last_obs - …"` it is computed over the full
data files, because it defines the split; for every other anchor
(`simulate { to
= last_obs + … }`, breakpoints, `value_at`) over the training
rows only, so appending held-out rows never changes a fit's end time or its
identity. A fit and a forecast print a note naming any stream whose last
observation is more than one of its own periods before `last_obs`, so a stream
that stopped reporting is visible.

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
- **The IR** already carries what the runtime needs: the calendar axis
  (`time_unit`, `origin`, `origin_rata_die`), each stream's `covers`, column
  roles, emit schedules and forcing knot times. This design changes no IR field;
  the calendar `covers` forms (D45) are the next IR change.
- **The runtime** owns everything about data, fits and forecasts. Each runtime
  type is built at one boundary, and the engines receive `f64` vectors. The CLI
  time grammar (`parse_time_spec`, `rust/crates/cli/src/fit/runner.rs:2162`)
  stays the runtime's one parser for times a user types, including
  `--horizon
  "4 weeks"`; its months and years are affine spans, as the IR's
  are, and it accepts no calendar-month horizon.

### 5.2 Invariants and who enforces them

"Static" means a violating value cannot be constructed; "checked" means a
constructor refuses it. No invariant is left to a comment.

| Invariant                                                                              | Owner                                 | Kind                          |
| -------------------------------------------------------------------------------------- | ------------------------------------- | ----------------------------- |
| A calendar duration shifts an instant only through `add_calendar_*`, in every position | `ocaml/lib/compiler/time_typing.ml`   | static                        |
| Instant arithmetic follows `docs/dates.md`, "Three concepts"                           | `time_typing.ml`                      | static                        |
| `date_range` calendar stepping is computed from its start                              | `ocaml/lib/compiler/expander.ml`      | static                        |
| A period has finite, increasing endpoints                                              | `Period::new`                         | checked (exists)              |
| A row's period is computed in one place                                                | `observed_rows`                       | static (only producer)        |
| Training is decided on periods, never labels                                           | `TrainingPartition::from_cutoff`      | static (no label input)       |
| Training rows are a prefix of each stream                                              | `TrainingPartition`                   | checked                       |
| A held-out score's fitted rows digest to the fit's prefix                              | `score` preflight                     | checked                       |
| Targets end after the origin, do not overlap, and stay within the horizon              | `ForecastPlan::new`                   | checked                       |
| Conditioning fields come from the fit record and the filter, not a caller              | `ForecastPlan` constructor visibility | static                        |
| A period is never moved to fit a step grid                                             | alignment check                       | checked (refusal)             |
| A time function outside its range is an error unless overridden                        | lookup clamp branch, run setup        | checked                       |
| A rendered date parses back to the same model time                                     | `CalendarAxis`                        | checked (property test)       |
| Every artifact carrying a model time carries the calendar axis                         | artifact constructors                 | static (constructor argument) |

### 5.3 Artifact contracts

Every artifact that carries a model time carries the `CalendarAxis` once, and
names each instant as `{t, date}`, with `date` recorded or explicitly "not
recorded" for a model without an origin.

- **Simulation record** (`run.json`): gains `calendar`.
- **Fit record** (`fit.meta.json`): `training_partition` replaces
  `training_window`; gains `calendar`.
- **Predictive artifacts** (`fit predict`, `simulate --init-state`): the plan
  (mode, forecast origin, horizon, both information sets, notes) in the JSON;
  per row in the TSV, a `role` (`trained`, `held_out`, `target`), the period
  (`t_start`, `t_stop`, and dates when the model has an origin), and for targets
  `horizon` and `lead`.
- **Score artifact** (`2026-11-14-score-artifact.md`): its lead-time type gains
  a fixed-origin variant carrying the forecast origin; each scored target
  records its horizon and lead; scores are kept per forecast date and horizon
  and never pooled across horizons (D35); the fit's conditioning travels with
  the score.

### 5.4 Identity

The training rule is part of a fit's identity: a `holdout_after` fit on a start-
or day-labelled stream trains on different rows after this change, so it must
re-key. The removal of `obs_alignment` from the fit identity, the score-artifact
proposal's row digests, and the removal of holdout files from fit identity
change the same structure (`FitDigest`) and land together in one `HASH_VERSION`
bump, from 3 to 4 (D47). A forecast plan's mode, origin, horizon and target list
are identity for any stored predictive or score result; its notes are
presentation.

### 5.5 Names

| Concept                                                                       | Name                                         | Not to be called               |
| ----------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------ |
| the calendar date of `t = 0`                                                  | origin                                       | —                              |
| the instant after which a forecast sees no data                               | forecast origin                              | "origin" alone                 |
| the date or number printed in a row's time column                             | label                                        | "time" of the row              |
| what a row measures                                                           | its instant or period; the code's `Measures` | —                              |
| the latest end among training rows                                            | training extent                              | `train_end` as a label         |
| the instant `holdout_after` names                                             | cutoff                                       | `score_from` as a user concept |
| a target's count of its stream's periods from the one under way at the origin | horizon                                      | "step"                         |
| how far ahead a forecast reaches, as a duration (`--horizon "4 weeks"`)       | horizon (on the command line)                | —                              |
| the duration from the forecast origin to a target's end                       | lead time                                    | —                              |
| the predictive mode flag of `fit predict`                                     | `--mode`                                     | `--horizon`                    |
| the count of scored rows in `compare`                                         | scored steps                                 | horizon                        |

## Part 6. Worked examples

**A day-labelled stream across a cutoff.** `covers = day(time)`, labels 1–16,
`holdout_after = 10`. Row 9 counts day 9 and ends at 10, so it trains; row 10
counts day 10 and ends at 11, so it is held out (today it trains, since 10 ≤
10). The training extent is 10. The fit reports "trained on 9 rows, through day
10".

**Two cadences.** A weekly `closing_at(time, 7 'days)` stream labelled 7–70 and
a daily `day(time)` stream labelled 0–69, `holdout_after = 24`. Daily rows 0–23
train (ends 1–24); weekly rows 7, 14 and 21 train (ends at or before 24). The
extent is 24. Weekly row 28, covering days 21–27, did not train and is scored,
like every daily row from 24 on.

**A monthly series.** Window-column rows ending at days 31, 59, 90, 120, 151 and
181, no `emit_schedule`. A forecast is refused: "stream 'cases' states its
periods per row and declares no `emit_schedule`; forecast targets need a
declared rhythm". Today `fit predict` predicts at 211, 241, … where the
month-ends are 212, 243, ….

**A forecast origin before conditioned data (invalid).** The state was filtered
through rows ending at day 40, and the user asks for a forecast origin of
day 35. Refused: "the forecast origin, day 35, is earlier than data the state
was conditioned on (through day 40)".

**A period moved by a grid (invalid).** PGAS, `simulate { from = 0.5 'days }`,
`dt = 1 'days`, a `closing_at(time, 2 'days)` stream with labels 3, 5, 7. The
first period, days 1–3, does not lie on the grid 0.5, 1.5, 2.5, …. Refused,
naming row 3 and the grid times 2.5 and 3.5. Today PGAS scores days 1.5–3.5 and
labels the row 3.

**A forcing past its data.** An interpolated temperature forcing with data
through day 100, a run to day 130: an error naming the forcing and the span
100–130, unless the override is given, in which case the run warns and records
it. A forecast from origin day 60 additionally records that the forcing's values
over days 60–100 were read after the forecast origin.

## Part 7. Deferred, and why this design does not block it

| Later work                                             | What it adds                                                                                            | Why nothing here is rewritten                                                                                            |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Calendar `covers` forms (D45, the immediate follow-up) | `month(time)`, `iso_week(time)`, `mmwr_week(time)`; a load-time check that labels fall on the right day | a calendar width is one more declared tiling (§4.3, case 2); one IR variant                                              |
| Aggregated targets                                     | weekly targets from a daily stream                                                                      | a target is a stream and a period; an aggregate is a new target kind beside it                                           |
| Observation identity across datasets                   | a key per row independent of position                                                                   | consumers read membership only through `RowPrefix` and `trained`; a key set can replace the count behind the same method |
| Data vintages and `as_of` reconstruction               | "the data as known at the forecast origin"                                                              | the two information sets are already separate; a vintage becomes one more field of `RowPrefix`                           |
| Hubverse import and export                             | external forecasts scored on camdl targets                                                              | a target's stream, end date and horizon are a hub's join key; the score artifact stores the hubs' 23 quantile levels     |
| A per-forcing extrapolation declaration                | `extrapolate = hold` in the model                                                                       | replaces the run-level override for that forcing; the error remains the default                                          |
