# Forecasting and scoring

How well a model fits the data it was estimated from overstates how well it
predicts new data, because the estimates adapt to the noise in those data as
well as to their signal. A model with enough freedom can fit its training data
closely and still forecast badly (Hyndman and Athanasopoulos 2021, §5.8). The
standard check is to hold back the last part of a series, fit the model to the
rest, forecast the part held back, and score the forecast against it. That score
estimates the model's accuracy on data it was not fitted to. Set beside the
in-sample score, it shows how much of the in-sample fit was overfitting, and
compared across models it ranks them by how well they forecast. A held-out score
on its own does not separate overfitting from a model whose dynamics are wrong,
since both forecast badly; the gap between the two scores is what points to
overfitting.

The score measures any of this only if nothing from the held-back rows reached
the fit or the forecast. A route by which they can is called _leakage_. This
page is for a developer changing how camdl splits data into training and
held-out rows, builds a forecast, or scores one. It describes the concepts, the
routes by which held-out data can reach a forecast, the types and checks that
close each route, and how a forecast is scored.

Some related subjects are covered on other pages and only linked from here:

- How a model time relates to a date, what instant or period a data row
  measures, and how the `covers` declaration maps a row's label to a period:
  [Dates and calendar time](../dates.md), under "The calendar axis", "Three
  concepts" and "What a data row means is not `date()`'s job".
- The files a prediction and a score write: the
  [run specification](../camdl-run-spec.md), §2.2.2 and §2.2.4.
- The scoring rules, and how to read a comparison of two models:
  [Model comparison](../methods/model-comparison.md).

Throughout this page the examples share one setup. The model counts time in days
from `origin = date("2025-01-05")`, a Sunday, so day 42 is the start of 16
February. A stream `weekly` counts cases per week. Its data file has ten rows,
each labelled with the Saturday that ends its week, from 11 January to 15 March.
The stream declares `covers = ending_on(time, 7 'days)`, so the row labelled 15
February counts the cases of 9 to 15 February. Some examples add a second
stream, `daily`, which counts cases per day and declares `covers = day(time)`,
so its row labelled 28 January counts the cases of that day.

## 1. Concepts

### 1.1 Training and test data

The available series is separated into training data, which are used to estimate
the parameters, and test data, which are used only to measure accuracy (Hyndman
and Athanasopoulos 2021, §5.8). camdl calls the test rows _held out_.

In a time series the separation has to follow time order. A forecast made in
practice has no observation later than the moment it is made, so an evaluation
measures the same thing only if every training row precedes every test row.
Hyndman and Athanasopoulos (2021, §5.10) state the requirement for their
cross-validation procedure: the training set "consists only of observations that
occurred prior to the observation that forms the test set", so that "no future
observations can be used in constructing the forecast". A test row chosen at
random has training rows on both sides of it. Its score measures how well the
model interpolates rather than how well it forecasts, and the parameters were
fitted to rows from after it.

The cutoff need not be near the end of the available data. Any past date gives a
retrospective forecast, often called a backtest: the model is fitted to the rows
before the date and scored on the rows after it. One cutoff gives one sample of
forecast performance. A cutoff chosen after seeing how forecasts from several
cutoffs scored has used the held-out rows to make a choice, which is a form of
the leakage described in §2.

The forecasting literature distinguishes two designs (Tashman 2000). A
_fixed-origin_ evaluation splits the series once and forecasts every test period
from the end of the training data. A _rolling-origin_ evaluation repeats the
split at successive times and forecasts from each; Hyndman and Athanasopoulos
(2021, §5.10) call it "evaluation on a rolling forecasting origin". camdl
implements only the fixed-origin design so far, and §5 lists the rolling origin
among what is not supported. A fit declares one split, and the split is always
at the tail of the data:

```toml
[data]
holdout_after = "2025-02-16"   # a date; or a model time (42); or "last_obs - 4 weeks"
```

The value is an [instant](../dates.md#three-concepts), the _cutoff_. A date used
as a cutoff is the instant that day begins, so `"2025-02-16"` is day 42. A row
is used for training if the end of what it measures is at or before the cutoff.
In the example the `weekly` rows labelled 11 January to 15 February are used for
training, and the four rows labelled 22 February to 15 March are held out. The
rule reads the end of the row's period and never its label; "Which rows train
under a holdout cutoff" in [Dates and calendar time](../dates.md) gives the
cases where the two differ. A fit can name separate holdout files under
`[data.holdout]` instead, and then every row of the training files is used for
training. With either form, the _training extent_ is the latest end among the
training rows of every stream; in the example it is day 42. A rolling-origin
evaluation can be done by running one fit per cutoff, and no camdl command does
that (§5).

### 1.2 Forecast origin, forecast targets and lead time

The _forecast origin_ is the instant after which a forecast uses no observation.
In camdl it is the training extent: day 42, the start of 16 February, in the
example. [`ForecastPlan::new`](rust:cli::fit::forecast_plan::ForecastPlan::new)
has no origin argument and reads the extent of the training partition it is
given. (Note that the model's `origin` declaration is a different thing, the
calendar date of day 0. Prose and messages write "forecast origin" in full; only
the fields and methods of the forecast modules shorten it to `origin`.)

A forecast predicts a list of _forecast targets_. A forecast target is one
period of one stream, of the same kind as the periods the stream's data rows
measure: a week for `weekly`, a day for `daily`. For a stream that reads the
state at instants, a target is an instant. A target exists whether or not an
observation of it exists. How many targets a forecast has therefore depends on
the interval at which each stream reports as well as on how far the forecast
reaches.

`--horizon` sets how far the forecast reaches. It is a duration, such as
`"4 weeks"` or `"28 days"`, and the forecast predicts every target of every
stream that ends within that duration of the origin
([`parse_reach`](rust:cli::fit::forecast_plan::parse_reach)). With
`holdout_after = "2025-02-16"` and `--horizon "4 weeks"`, the `weekly` stream
has four targets:

| Forecast target        | Periods ahead | Lead time |
| ---------------------- | ------------- | --------- |
| week of 16–22 February | 1             | 7 days    |
| week of 23 Feb – 1 Mar | 2             | 14 days   |
| week of 2–8 March      | 3             | 21 days   |
| week of 9–15 March     | 4             | 28 days   |

A `daily` stream bound to the same fit would have 28 targets, one for each day
from 16 February to 15 March, with lead times of 1 to 28 days. The reach only
truncates each stream's list of targets. It does not move the origin and it does
not define a period. A reach in months or years is refused for a model with a
calendar `origin`, because a month counted from a forecast origin that is known
only at run time has no fixed length; write it in weeks or days. Without an
`origin`, a month is a fixed span of 30.44 days, so `--horizon "1 month"` would
give the `weekly` stream the same four targets.

Each target carries two measures of how far ahead it is.
[`Target`](rust:cli::fit::forecast_plan::Target) holds both, with the stream and
the period.

- The _lead time_ is the duration from the forecast origin to the end of the
  target, in model time units. It is comparable across streams.
- _Periods ahead_ counts a stream's targets from 1, in the stream's own periods:
  the first target that ends after the origin is 1 period ahead. A target 3
  periods ahead is three weeks ahead for `weekly` and three days ahead for
  `daily`, so the count is comparable only within one stream. It is the
  `periods_ahead` field of `Target` and column of a forecast's output.

<!-- figure: forecasting-origin-and-targets, caption: One forecast of the `weekly` stream under `holdout_after = "2025-02-16"` and `--horizon "4 weeks"`. -->

```
week ending       11 Jan … 8 Feb  15 Feb |  22 Feb   1 Mar   8 Mar  15 Mar
                [---- six training rows --]|[  1  ] [  2  ] [  3  ] [  4  ]
training rows     <----------------------->|
conditioning rows <----------------------->|
                          forecast origin, 16 Feb (day 42)
periods ahead                                  1       2       3       4
lead (days)                                    7      14      21      28
```

<!-- /figure -->

Hyndman and Athanasopoulos (2021, §5.8) use "forecast horizon" for the number of
steps ahead, _h_, with the first future period at _h_ = 1. For a stream whose
first target is a whole period, their _h_-step forecast is the target whose
`periods_ahead` is _h_.

The epidemic forecast hubs define a horizon by dates. In the CDC COVID-19
Forecast Hub and in FluSight, a forecast has a `reference_date`, the Saturday
that ends the week in which it is submitted, and each prediction has
`target_end_date = reference_date + horizon × 7 days`, with horizons from −1 to
3 (CDC COVID-19 Forecast Hub; FluSight Forecast Hub). Horizon 0 is the week of
submission, and horizon −1 is the week before it, for which data usually exist.
The earlier US COVID-19 Forecast Hub named the unit in each target, with "N wk
ahead" and "N day ahead" targets in one file (US COVID-19 Forecast Hub).

camdl's periods ahead differs from a hub horizon in two ways. First, it is
counted from the end of the data the fit trained on, and a hub's horizon from
the submission date, so the two agree only when the data are current. The CDC
hub's README gives an example of its data timing: the release of 11 June 2025
holds data through the week ending 7 June. Forecasts due that day have the
reference date 14 June by the hub's rule, so their horizon 0 is the week of 8 to
14 June. A camdl forecast trained through 7 June has the same week 1 period
ahead, so when the data are current, camdl's periods ahead is the hub's horizon
plus 1. If the same stream's data stopped a week earlier, the target 1 period
ahead would be the hub's horizon −1. Second, camdl's count starts at 1, because
every target ends after the origin. A forecast converted to a hub format needs
its hub horizon computed from each target's end date and the hub's reference
date; camdl writes the end date and does not write a hub horizon.

### 1.3 What a forecast is conditioned on

A forecast from a state-space model depends on two unknowns that are estimated
from the data: the model's parameters, and its latent state at the forecast
origin, such as the number of people infectious at the start of 16 February.
Each estimate is informed by some rows of data, and the two sets of rows need
not be the same.

The _training rows_ are the rows the parameters were fitted to. The
_conditioning rows_ are the rows the latent state at the origin was conditioned
on. To condition the state on rows is to run a particle filter through them, so
that the particles at the origin approximate the distribution of the state given
those rows.

camdl records the two separately because they answer different questions about a
prediction. Whether the parameters were fitted to a row decides whether that
row's score is in-sample. Whether the state was conditioned on the rows before a
row decides how far ahead the prediction of it is. A prediction of the week of 2
to 8 March from a state conditioned on every row through 1 March is a forecast
one week ahead, even though the parameters were fitted only through 15 February.
It is not a forecast three weeks ahead from 16 February. Both sets of rows are
recorded per stream as a row count and a digest of those rows
([`RowPrefix`](rust:cli::fit::training_partition::RowPrefix)), and
`forecast.json` and `predictive.json` write them as `training_rows` and
`conditioning_rows`.

For a fixed-origin forecast the two hold the same rows. The forecast driver
filters through the training rows and checks that the rows it was given are the
training rows the fit recorded (§3.5). A state conditioned on rows newer than
the fit, with the parameters left as fitted, would make the conditioning rows
the larger set; camdl does not do this in a forecast (§5).

### 1.4 Kinds of prediction

Not every prediction camdl makes is a forecast. Four properties describe a
prediction, and each is recorded under its own name in the artifacts:

- the mode, which says which rows the state was conditioned on;
- the role of each predicted row, which says whether the parameters were fitted
  to it;
- the treatment of the parameters: a posterior, or a single point;
- the simulation backend, which limits which modes exist.

**Mode.** The mode is the value of `camdl predict --mode`
([`PredictModeArg`](rust:cli::args::types::PredictModeArg)).

- `free_forward` conditions on no rows. The model runs from the start of the
  run, and the data reach the prediction only through the parameters.
- `one_step` predicts each data row from a state filtered through every earlier
  row of every stream. Under `camdl predict` the filter walks the fit's stored
  training rows. Under `camdl score` it carries on through the outcome rows,
  predicting each one and then assimilating it.
- `forecast` conditions on the training rows up to the forecast origin and on no
  row after it. Passing `--horizon` without `--mode` selects it.

A `forecast`, and a `one_step` prediction of a held-out row, are forecasts in
the sense of §1.1. A `one_step` prediction of a training row is what Hyndman and
Athanasopoulos (2021, §5.3) call a fitted value. They note that "fitted values
are often not true forecasts", because the parameters were estimated from all
the observations, including those after the predicted one. A `free_forward`
prediction over the training rows is in-sample in the same way.

**Role.** Whether the parameters were fitted to the predicted row is a property
of each row, independent of the mode. It is the `role` column
([`RowRole`](rust:cli::fit::predict::RowRole)): `trained`, `held_out`, `future`
for a period the prediction read no data row for, or `unknown` for a parameter
point with no record of its training rows. "In-sample" is a value of this
property and not a mode.

**Parameter treatment.** How the parameters enter is recorded as
[`TreatmentKind`](rust:cli::fit::predict::TreatmentKind). A `posterior`
prediction runs once per posterior draw and pools the results, so its intervals
include parameter uncertainty. A `plug_in` prediction uses one point, an
optimizer's winner or a point given in a file, and its intervals include process
noise and observation noise only. They therefore cover the observed values less
often than their nominal level when the parameters are poorly identified. A
plug-in forecast is still a forecast, and a proper score evaluates it as one; it
is likely to score worse than a posterior forecast of the same model for this
reason.

**Backend.** Two backends fit models. On the chain-binomial backend the state is
random given the parameters, so conditioning it on data changes the prediction,
and all three modes exist. On the ODE backend the state is a deterministic
function of the parameters, so conditioning it on rows adds nothing the
parameters do not already determine. A forecast of an ODE fit is still well
defined: past the origin it is the trajectory each draw determines, read over
the targets, with observation noise added. camdl does not implement it yet. A
fixed-origin forecast of an ODE fit is refused by
[`refuse_unsupported`](rust:cli::fit::forecast_output::refuse_unsupported), and
the refusal points to `--mode free_forward`. The Gillespie backend simulates but
does not fit, and a forecast on it is refused as well.

The inference method matters only through what it produced. PMMH and PGAS
produce posterior draws on the chain-binomial backend, and IF2 produces one
winning point on the same backend. MH and the NLopt optimizers fit on the ODE
backend.

| What the fit produced                    | Backend        | Fixed-origin forecast        |
| ---------------------------------------- | -------------- | ---------------------------- |
| posterior draws (PMMH, PGAS)             | chain-binomial | yes, `posterior`             |
| one winning point (IF2)                  | chain-binomial | yes, `plug_in`               |
| a parameter point in a file, with data   | chain-binomial | yes, `plug_in`               |
| posterior draws (MH)                     | ODE            | refused; not implemented yet |
| one winning point (the NLopt optimizers) | ODE            | refused; not implemented yet |

The supported and refused cells are tested in
`rust/crates/cli/tests/fit_predict_forecast.rs`.

### 1.5 Where forecast targets come from

A forecast target is one period of one stream (§1.2), and a stream's targets
continue the rhythm in which it reports: the targets of `weekly` are the weeks
after its last training row. camdl reads that rhythm from what the stream
declares, never from the spacing of its past rows. In camdl, a stream gets its
targets in one of three ways:

- from a uniform `covers` form, which tiles time into periods of one width;
- from an `emit_schedule`, which lists the times at which the stream reports;
- from its held-out rows, for a stream with `window_start` and `window_stop`
  columns and no schedule.

[`stream_targets`](rust:cli::fit::forecast_plan::stream_targets) is the only
function that makes targets, and
[`TargetSource`](rust:cli::fit::forecast_plan::TargetSource) records which of
the three a stream used. The rules for each follow.

A stream with a uniform `covers` form, such as the two example streams, takes
its targets from the declared tiling. Its periods tile without gaps, so its last
training row fixes the phase, and the targets are the periods that follow at the
declared width. Each period is computed from that row and the step count, so no
rounding accumulates over a long reach.

A stream with an `emit_schedule` and no uniform `covers` form takes its targets
from the schedule: the periods between consecutive schedule times for a stream
with `window_start` and `window_stop` columns, or the schedule's instants for a
stream that reads the state. Where a stream has both an `emit_schedule` and
periods implied by its data, the two must describe the same periods over the
reach. A data period that overlaps a schedule period without being equal to it
is refused, because there is no rule for which one to follow
([`check_schedule_agrees`](rust:cli::fit::forecast_plan::check_schedule_agrees)).

A stream with window columns and no `emit_schedule` declares no rhythm. When it
has held-out rows, their periods within the reach are its targets, after a check
that they are in order, distinct, and overlap neither each other nor a training
row. Nothing is extrapolated. Such a target's periods ahead counts held-out rows
and is not comparable with the periods ahead of a tiled stream's target; compare
these targets by lead time.

Anything else is refused, per stream
([`StreamRefusal`](rust:cli::fit::forecast_plan::StreamRefusal)). A
window-column stream with no schedule and no held-out rows, and a stream that
reads the state at instants with no schedule, are refused because the spacing of
past rows is not a declaration. A monthly series is the usual case: calendar
months cannot be written as a uniform `covers` form, so the stream has window
columns, and without an `emit_schedule` a forecast of it is refused with a
message naming the two ways to declare a rhythm. A stream that binds no rows is
refused because its tiling has no phase. The plan fails as a whole only when no
stream has a target.

When a stream has a declared rhythm, held-out rows only select which targets
have an observed value. They do not renumber the targets, so a missing week does
not shift the periods ahead of the weeks after it. A held-out row within the
reach whose period is not one of the stream's targets is refused. Skipping it
would leave a period out of the score without saying so.

`camdl predict --mode free_forward` builds no plan. Past a stream's data its
rows continue at the most common gap between the stream's observation times
([`forecast_times`](rust:cli::fit::predict::forecast_times)). That rule is a
plotting grid for an unconditioned projection, and it is never a source of
forecast targets.

### 1.6 Forecast targets that began before the origin

When a fit has streams that report at different intervals, their last training
rows usually end at different times. The forecast origin is the latest of those
ends, so a stream that reports less often can have a period already under way at
the origin. Take both example streams with `holdout_after = "2025-01-29"`:

- the `daily` rows are used for training through 28 January;
- the `weekly` rows are used for training through the week ending 25 January;
- the forecast origin is the start of 29 January, day 24, set by the `daily`
  stream.

The `weekly` stream's next week, 26 January to 1 February, began three days
before the origin. Because the stream's targets tile from its last training row
(§1.5), that week is its first target, 1 period ahead with a lead time of 4
days.

<!-- figure: forecasting-two-cadences, caption: The `daily` and `weekly` streams under `holdout_after = "2025-01-29"`. Each stream counts periods ahead in its own periods. -->

```
date          19 … 25 Jan | 26  27  28 Jan | 29 Jan … 1 Feb | 2 … 8 Feb | 9 … 15 Feb
daily         observed through 28 Jan       | 1, 2, 3, 4 … periods ahead
weekly        [ 19–25 Jan ] [ happened      | simulated     ] [ 2–8 Feb ] [ 9–15 Feb ]
                            [----- 1 period ahead ----------] [   2     ] [   3      ]
                                       forecast origin, 29 Jan (day 24)
```

<!-- /figure -->

The target is predicted whole: the forecast of it is a draw of the count for all
seven days. The cases of 26 to 28 January have happened, but the `weekly` stream
has not reported them, so they are a latent quantity, estimated through the
conditioning on the `daily` rows. The cases of 29 January to 1 February are
simulated. A forecaster on a hub faces the same kind of target when the week of
submission is partly over; FluSight calls its horizon-0 prediction "a nowcast
for the week of submission" (FluSight Forecast Hub).

#### On the chain-binomial backend

The count for such a week is the per-stream running total each particle carries
([`ParticleState::acc`](rust:sim::inference::types::ParticleState)). The total
is copied with the particle on resampling and is reset only where a declared
period of its stream starts. The forecast driver declares every target as a
period of its stream before the filter runs, so at the end of the week the total
holds that particle's count for 26 to 28 January, as filtered, plus its
simulated count for 29 January to 1 February. If the week were not declared, the
total would not be reset at its start and would include the week before as well.

The test `the_week_under_way_matches_the_importance_sampling_reference` in
`rust/crates/cli/src/fit/forecast_driver_tests.rs` checks this at fixed
parameters. It compares the forecast of the week with an estimate of the same
week's count given the daily rows that uses no particle filter (simulated
trajectories weighted by the likelihood of the daily rows), and checks that
leaving the week undeclared gives a different answer.

A restart is a different case. `simulate --init-state` restores compartment
counts and not the running totals, so it cannot write the row of a period that
began before the restart.
[`refuse_period_under_way`](rust:cli::obs_emit::refuse_period_under_way) refuses
such a restart. A forecast does not go through a saved state: it filters from
the start of the run, so its totals are always present.

#### On the ODE backend

The week's count would be the difference of the cumulative flow at the week's
two ends on each draw's trajectory. A fixed-origin forecast on this backend is
refused (§1.4).

### 1.7 Forcings and covariates after the origin

A model may read values that it does not simulate, from an external data file or
defined in the model itself. Examples are a forcing interpolated from a data
table, such as temperature or a vaccination schedule, and a column of the
observation data that the likelihood reads, such as the number tested in a
survey. For a forecast these raise the question the forecasting literature puts
as ex-ante against ex-post. An _ex-ante_ forecast uses "only the information
that is available in advance", and an _ex-post_ forecast uses "later information
on the predictors"; of ex-post forecasts, Hyndman and Athanasopoulos (2021,
§7.6) write that "these are not genuine forecasts, but are useful for studying
the behaviour of forecasting models". A third case is a scenario, in which the
future values of a predictor are assumed.

A forecast that silently read realised future values of a covariate would score
better than any forecast that could have been made at the origin, which is a
form of leakage (§2). camdl cannot tell from a table whether its rows after the
origin were planned, assumed or observed later. A forecast that reads a table
forcing after its origin is therefore refused before any filter runs, until each
such forcing has a statement
([`CovariateStatement`](rust:cli::args::types::CovariateStatement)):

- `--covariate NAME=prospective` says the values were fixed or assumed at the
  origin. The table is read as it is.
- `--covariate NAME=realised` says they were observed after the origin. The
  table is read as it is, and the forecast and every score of it are labelled
  "conditional on covariates observed after the origin".
- `--covariate NAME=hold` carries the table's value at the origin forward and
  does not read the table after it.

The statements are part of the prediction's identity and are written to
`forecast.json`. Which forcings count as read after the origin, by interpolation
method and lag, is specified in the run specification, §2.2.2, "Forcings read
after the origin".

The statement is a declaration and camdl does not check it. `prospective` on a
table of realised values produces an ex-post forecast labelled as ex-ante.

One dependence is reported and not closed. A linearly interpolated table has a
value between its last row before the origin and its first row after it that
depends on both rows, and a cubic-spline table's value at every time depends on
every row. The filter reads such a forcing before the origin, so rows after the
origin reach the state at the origin.
[`pre_cut_dependence`](rust:cli::fit::forecast_covariates::pre_cut_dependence)
finds these forcings, the forecast prints a note for each, and `forecast.json`
records them under `covariates.interpolation`. The forecast is not refused.

Design columns follow the same division. A stream whose likelihood reads a data
column has no value of it for a target past the data, so the stream is refused a
forecast unless a design supplies the column. `--design SOURCE=FILE` supplies it
from a file written at the origin, which makes a prospective forecast.
`--realised-design` names the held-out rows' own values, which makes an ex-post
evaluation ([`DesignMode`](rust:cli::fit::forecast_plan::DesignMode)).
`camdl predict` refuses that flag for a stored fit, which holds no held-out
values, and for a parameter point given in a file. A target with no design value
is refused and never filled in.

A separate rule covers a forcing evaluated outside the range on which it is
defined: a data table before its first or after its last row, or a B-spline
outside its window. That is an error in every command, because holding the end
value would rest a forecast on a number nobody supplied. The override is
`--allow-forcing-extrapolation`, under which the forcing holds its end value and
the run warns once per forcing and side. The engine records that a side of a
forcing's range was left ([`forcing_range`](rust:sim::forcing_range)), and the
command decides whether that is an error or a warning
([`forcing_extrapolation`](rust:cli::forcing_extrapolation)). The check reads
the times the run evaluated, with the forcing's `lag` applied, so a forcing read
only under a condition that is false past its range raises nothing. The override
is part of a run's identity. A forecast that runs past the model's
`simulate { to }` extends the run and the intervention plan to its latest target
([`extend_run`](rust:cli::fit::forecast_driver::extend_run)) and extends no
forcing.

### 1.8 `last_obs`

`last_obs` is the latest end of a row that carries an observed value, and
`first_obs` the earliest start, taken across every bound stream
([`observed_window`](rust:cli::fit::training_partition::observed_window)). An
`NA` row counts toward neither.

The two anchors are computed over different rows depending on where they are
used, and the difference is what keeps held-out rows from moving a fit. In
`holdout_after = "last_obs - 4 weeks"` they are computed over the whole data
files, because they define the split
([`resolve_holdout_cutoff`](rust:cli::fit::runner::resolve_holdout_cutoff)).
Everywhere else, in `simulate { to = last_obs + 2 'weeks }`, in a breakpoint or
in `value_at`, they are computed over the training rows only. Appending held-out
rows to a data file therefore changes no anchor of the fitted model, and so
changes neither the end of its run nor its identity.

When a fit starts it prints a note naming any stream whose last observation is
more than one of its own periods before `last_obs`
([`partition_lines`](rust:cli::fit::runner::partition_lines)), so a stream that
stopped reporting is visible.

## 2. Data leakage

Kaufman, Rosset and Perlich (2011) define leakage as the introduction of
information about the target of a prediction that should not be legitimately
available. For prediction in time they name the rule the _no-time-machine
requirement_: a legitimate model builds only on information from a time no later
than that of the target (§3.2). Their remedy is a _learn-predict separation_,
which in the common case is done by time-stamping every observation and cutting
at one point in time (§4.1). Kapoor and Narayanan (2023) survey leakage in
published science and list eight kinds. Two of them apply directly to a forecast
from a fitted model: a test set that contains data from before the training set,
which they call temporal leakage, and a model that uses features that should not
be legitimately available to it.

camdl's time stamp is the end of what a row measures, and its cut is the
training extent. A forecasting system built on that cut still has several routes
by which a held-out row, or something derived from one, can reach a forecast or
its score. The table lists each route and what closes it, using four kinds of
closure, which give different guarantees (§3.1):

- _shape_: a value that breaks the rule cannot be represented by the type;
- _constructor_: the type's fields are private and its only constructors refuse
  a value that breaks the rule;
- _command_: a command refuses the input at run time;
- _convention_: the rule holds because one function is the only one written to
  decide it, and nothing stops a second.

| Route                                                          | Closed by                                                                                                                                                       | Kind                        |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| A held-out row among the rows the parameters were fitted to    | the training rows of a stream are stored as a row count                                                                                                         | shape                       |
| A training row later than a held-out row                       | `from_cutoff` refuses training rows that are not a prefix; `check_holdout_files` refuses a holdout row that ends at or before the training extent               | constructor, command        |
| The split decided from a row's label                           | `trains_under_cutoff` reads the end of what the row measures                                                                                                    | convention                  |
| A later command cutting the data differently                   | the partition is read from the fit's record with `from_record` and never cut again                                                                              | constructor                 |
| A fitted row edited after the fit                              | `from_record` compares the digest of each stream's training rows                                                                                                | constructor                 |
| A held-out row conditioning the state at the origin            | a target's value is an `ObsCell::HeldOut`, which the likelihood never assimilates; `conditioning` compares the rows handed to the filter with the training rows | shape, constructor          |
| A forecast origin other than the end of the training rows      | `ForecastPlan::new` takes no origin                                                                                                                             | shape                       |
| A target that overlaps a training row                          | `ForecastPlan::new` refuses it in every build, failing the whole forecast                                                                                       | constructor                 |
| Outcomes scored against a different series                     | `check_agreement` compares the outcome file's leading rows with the conditioning rows                                                                           | command                     |
| A score that changed the forecast it scores                    | the prediction is computed again with the outcomes bound and compared with the stored one, byte for byte                                                        | command                     |
| A held-out score labelled in-sample                            | `camdl score` refuses a stored prediction whose manifest does not record its training rows                                                                      | command                     |
| Realised covariates read after the origin                      | refused until a `--covariate` statement is given; `realised` is then a label                                                                                    | command, then a declaration |
| Realised design values                                         | refused without `--design` or `--realised-design`; ex-post is then a label                                                                                      | command, then a declaration |
| Rows after the origin moving an interpolated forcing before it | a note and a record                                                                                                                                             | not closed                  |
| A held-out value in a stored fit                               | the stored fit keeps each held-out row's file, position and period, and not its value                                                                           | by what is stored           |
| The model chosen after looking at the held-out rows            | nothing                                                                                                                                                         | not closed                  |

The last row is outside what software can check.
[Model comparison](../methods/model-comparison.md), §6.3, states the same limit
for a held-out score.

The steps below follow one forecast of the example from the data file to its
score and say where each row of the table applies.

<!-- figure: forecasting-type-flow, caption: The values a forecast passes through. Each is built from the one above it. -->

```
ObservedRow            a data row and what it measures
     |
TrainingPartition      per stream, a row count and a digest
     |
fit.meta.json          the partition, read back and checked against the data
     |
ForecastPlan, Target   the origin and what is predicted
     |
StateConditioning      the rows the filter was given
     |
camdl score            outcome rows bound to targets
```

<!-- /figure -->

1. [`observed_rows`](rust:cli::fit::observed_rows::observed_rows) reads the
   `weekly` file once and returns ten rows, each with the period it measures.
   Every later step works from these rows.
2. The fit builds a `TrainingPartition` under `holdout_after = "2025-02-16"`.
   For `weekly` it holds the count 6 and a digest of the first six rows. A row
   trained exactly when its position is below 6, so within one file no row is on
   both sides and the training rows have no gap.
3. The fit writes the partition to `fit.meta.json`. Its stored problem keeps the
   six training lines, and for rows 7 to 10 keeps the file, position and period
   of each without its value.
4. `camdl predict` does not apply the cutoff again. It reads the count 6 from
   the record, digests the first six rows of the data it has, and refuses the
   data if the digest differs or the file has fewer than six rows.
5. `ForecastPlan::new` takes that partition and the reach. The origin is the
   partition's extent, day 42. The targets are the four weeks ending 22 February
   to 15 March.
6. The forecast driver digests the rows it was handed, fails unless they are the
   six training rows and all end at or before day 42, and only then records them
   as the conditioning rows. It appends the four targets to the stream as
   periods with no assimilated value.
7. `camdl score` is given a file with all ten rows. It refuses the file unless
   its first six rows equal, by digest, the rows the forecast conditioned on.
   Rows 7 to 10 supply the observed values of the four targets, matched by
   period.

## 3. The types

The types below are private to the `cli` crate, except `ObsCell`, which is in
`sim`. A forecast's values pass through them in this order:

```text
data file ──observed_rows()──► ObservedRow { label, measures: Measures, cell: Option<ObsCell>, .. }
                                    │
             TrainingPartition::from_cutoff / whole / from_record
                                    ▼
             TrainingPartition { rule, extent, streams: Vec<RowPrefix { stream, rows, digest }> }
                                    │
             ForecastPlan::new(training: TrainingPartition, horizon, streams)    no origin argument
                                    ▼
             ForecastPlan { origin = training.extent(), targets: Vec<Target>, .. }
                                    │
             forecast_driver::conditioning(plan, training streams)               private function
                                    ▼
             StateConditioning: the conditioning rows, with the time they end
                                    │   targets appended as ObsCell::HeldOut(v), or with no cell
                                    ▼
             particle filter: only an ObsCell::Scalar is assimilated
```

§3.1 says which of these types close a route of §2 by their shape, which by
their constructors, and which routes rest on convention. §3.2 to §3.5 give each
type: what it represents, the invalid state it removes, who builds it and who
reads it.

### 3.1 How the types close each route

**Shape.** Three routes are closed because a value that breaks the rule cannot
be represented.

A stream's training rows are stored as a prefix length, a `usize`, with a digest
of the rows it counts:

```rust
pub(crate) struct RowPrefix { pub stream: String, pub rows: usize, pub digest: RowDigest }
```

A row is a training row exactly when its position is below `rows`. No set of row
positions is stored anywhere, so training rows with a gap, or a row that is both
a training row and held out, cannot be written down.

The forecast origin is not an argument:

```rust
pub(crate) fn new(parameters: TrainingPartition, horizon: f64, streams: &[PlanStream])
    -> Result<ForecastPlan, PlanError> {
    let origin = parameters.extent();
```

`ForecastPlan`'s fields are private and `new` is its only constructor, so no
caller can start a forecast anywhere but the end of the training rows.

A held-out value has its own variant, and the likelihood matches on the pair of
the value's variant and the use it is being put to:

```rust
pub enum ObsCell { Scalar(f64), HeldOut(f64) }

match (cell, use_) {
    (Some(ObsCell::Scalar(v)),  LikelihoodUse::Assimilate)   => v,
    (Some(ObsCell::HeldOut(v)), LikelihoodUse::ScoreHeldOut) => v,
    (Some(ObsCell::HeldOut(_)), LikelihoodUse::Assimilate)
    | (Some(ObsCell::Scalar(_)), LikelihoodUse::ScoreHeldOut) => /* contributes nothing */,
```

`Assimilate` is the use that weights particles. A held-out value contributes
nothing to it, and making it contribute requires editing this match.

**Constructor.** Most of the remaining guarantees come from types whose fields
are private and whose constructors check the rule. Every value of the type that
exists passed the check. The check is code, so a change to it can weaken the
guarantee without a compile error.

| Rule                                                                                                        | Checked by                                                  |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| The training rows are a prefix in time                                                                      | `from_cutoff`, through `cutoff_prefix`                      |
| The rows read now are the rows the fit was trained on                                                       | `from_record`, by each stream's row count and digest        |
| Each target ends after the origin and within the reach; one stream's targets are ordered and do not overlap | `ForecastPlan::new`                                         |
| No target overlaps a training row of its stream                                                             | `ForecastPlan::new`, in every build                         |
| The filter was given exactly the training rows, all ending at or before the origin                          | `conditioning`, the only constructor of `StateConditioning` |

**Convention.** These rules hold because one function decides them, and nothing
would stop a second function from deciding them differently:

- The split reads the end of what a row measures. `ObservedRow.label` is a
  public field, and the rule is kept by every caller using
  `trains_under_cutoff`.
- A target's `observed` field is filled only from the rows the caller passes as
  held out. The plan does not check those rows against the partition.
- No path in the forecast driver makes an `ObsCell::Scalar` for a target. The
  match above makes a `HeldOut` value harmless; it does not stop a caller from
  wrapping a held-out value as a `Scalar`.

### 3.2 Observed rows

<!-- decl: rust:cli::fit::observed_rows::ObservedRow -->

<!-- decl: rust:cli::fit::observed_rows::Measures -->

An `ObservedRow` is one data row of one stream with the instant or period it
measures. A _stream_ here is one concrete observation series; the code calls it
a _leaf_ when an indexed observation block expands into several.

The type exists so that training, scoring and forecasting read what a row
measures and not the label in its time column. The same label names different
periods under different `covers` forms ([Dates and calendar time](../dates.md),
"What a data row means is not `date()`'s job"), so a rule that compared labels
with a cutoff would give different splits for the same data.

The type does not enforce this. `label` is a public field, and it is read where
the file's own time is the answer: writing the time column back, the time column
of `camdl predict`'s tables, and naming a row in a refusal. The cutoff rule is
held by one function,
[`trains_under_cutoff`](rust:cli::fit::training_partition::trains_under_cutoff),
which reads `measures.end()`.

`observed_rows` is the only reader of a stream's rows. The likelihood loader,
the training partition, the row digest and the forecast plan are all built from
its output, so what is scored and where it is scored come from one read of the
file. The binder requires each stream's ends to increase strictly, so a row's
position orders its stream in time.

### 3.3 `TrainingPartition`

<!-- decl: rust:cli::fit::training_partition::TrainingPartition -->

<!-- decl: rust:cli::fit::training_partition::RowPrefix -->

<!-- decl: rust:cli::fit::training_partition::PartitionRule -->

A `TrainingPartition` records which rows of each stream the parameters were
fitted to. It belongs in a page on forecasting because the forecast origin is
its extent and its list of prefixes is the training rows.

Because a stream's training rows are stored as a count, a partition cannot
represent a row that both trained and was held out, or training rows with a gap
between them. Membership is `position < rows(stream)`, and no reader compares a
time with a cutoff. The digest covers exactly the counted rows
([`row_digest`](rust:cli::fit::training_partition::row_digest)): each row's time
cells as written, its value or `NA`, and its auxiliary values. Rows appended
after the prefix do not change it, and an edit to a counted row does.

The fields are private and there are three ways to build one.

- [`from_cutoff`](rust:cli::fit::training_partition::TrainingPartition::from_cutoff)
  applies `holdout_after`. A row trains if the end of what it measures is at or
  before the cutoff, within `1e-9` (`TIME_TIE_EPS`), so that a date converted to
  model time cannot move a row across the cut. It checks at run time that the
  rows that train are a prefix. It refuses a split that leaves some stream with
  no observed training row, or that holds out no observed row. An `NA` row keeps
  its position in the prefix and is hashed into the digest, and it does not
  count as an observed row for either refusal.
- [`whole`](rust:cli::fit::training_partition::TrainingPartition::whole) trains
  every row, for a fit with no holdout or with `[data.holdout]` files. Under
  `[data.holdout]` the held-out rows are in files the partition does not
  describe, and
  [`check_holdout_files`](rust:cli::fit::runner::check_holdout_files) refuses a
  holdout row that ends at or before the training extent.
- [`from_record`](rust:cli::fit::training_partition::TrainingPartition::from_record)
  rebuilds the partition a fit recorded, against rows read now. It refuses a
  record written under another `PARTITION_RULE_VERSION`, a stream with no
  recorded prefix, a file with fewer rows than the prefix, a digest that
  differs, and a recorded extent that differs from the one the rows give.

`fit run` and `profile` build partitions through the first two. `camdl predict`
reads one through
[`read`](rust:cli::fit::training_partition::TrainingPartition::read) in every
mode, and a predicted row's `role` is its position against its stream's prefix.
`pfilter --save-final-state` cuts its streams with the same function
`from_cutoff` uses
([`cutoff_prefix`](rust:cli::fit::training_partition::cutoff_prefix)).

A parameter point given in a file has no fit record.
`camdl predict MODEL --holdout-after` builds a partition from the flag
([`RecordedPartition::Given`](rust:cli::fit::training_partition::RecordedPartition)),
and the result is a declaration that nothing verifies: a score of such a
prediction is labelled `hold_out_declared` and never `hold_out_tail` (§4.4).

The partition is written to `fit.meta.json` and copied as `training_rows` into
`predictive.json` and `forecast.json`. The fields are in the run specification,
§2.2.1.

### 3.4 `ForecastPlan` and `Target`

<!-- decl: rust:cli::fit::forecast_plan::ForecastPlan -->

<!-- decl: rust:cli::fit::forecast_plan::Target -->

A `ForecastPlan` is what one fixed-origin forecast predicts: the origin, the
reach, the training rows, the targets, where each stream's targets came from,
and each stream that was refused. Only a fixed-origin forecast builds one.

The plan removes three invalid states. A forecast cannot start anywhere but the
end of the training rows, because `new` takes the partition and no origin. Two
commands cannot disagree about a stream's targets, because `stream_targets` is
the only function that makes them (§1.5). A period under way at the origin
cannot be dropped, because the first target of a tiled stream is the first
period that ends after the origin, whenever it began.

`new` is the only constructor and the fields are private. After building the
targets it asserts, in every build, that each target ends after the origin and
within the reach, and that one stream's targets increase and do not overlap. A
failure of these is an internal error.

`new` also refuses a target that overlaps a training row of its stream, in every
build
([`PlanError::TargetOverlapsTraining`](rust:cli::fit::forecast_plan::PlanError)).
The property already follows from how each source builds its periods: a tiling
continues from the stream's last training row; a schedule period that cuts
across a row is refused by `check_schedule_agrees`; held-out windows that
overlap a training row are refused by their validation; and every target is
filtered to end after the origin, which is at or after the end of every training
row. No input reaches the refusal today. It is reported as an internal error and
fails the whole forecast, since dropping the one stream would hide a broken
construction as a smaller forecast.

A target's `observed` field names the held-out row that measures the same
period, by file and position. It is filled only from the rows the caller passed
as held out, and only from those that end after the origin. That those rows lie
after the training prefix is the caller's convention; the plan does not check it
against the partition.

A target's `design` holds the design values its stream's likelihood reads
(§1.7). They change the prediction, so they are part of the target set's digest
([`targets_digest`](rust:cli::fit::forecast_plan::ForecastPlan::targets_digest)),
which with the design mode and the covariate statements is the plan's
contribution to the prediction's identity
([`ForecastKey`](rust:cli::fit::predict::ForecastKey)). Held-out outcomes are
not in the digest. How an identity becomes a stored result's address is in
[Run identity](run-identity.md).

[`run_forecast`](rust:cli::fit::forecast_output::run_forecast) builds the plan,
and the forecast driver and the writer of `forecast.json` read it.

### 3.5 `StateConditioning` and the forecast driver

<!-- decl: rust:cli::fit::forecast_driver::StateConditioning -->

A `StateConditioning` records the rows the state at the origin was conditioned
on. It has one kind, for a particle filter run through a prefix of each stream.

Its field is private to the forecast driver's module, and its only constructor
is the private function
[`conditioning`](rust:cli::fit::forecast_driver::conditioning), which builds it
from the rows handed to the filter. The function digests those rows and fails
unless each stream's count and digest equal the plan's training rows and every
row ends at or before the origin. No caller passes conditioning rows in, so the
recorded conditioning rows are the rows the filter was given. The failure is
reported as an internal error, since a caller that reaches it has cut the
streams wrongly.

[`fixed_origin_forecast`](rust:cli::fit::forecast_driver::fixed_origin_forecast)
is the driver. It takes the plan, each stream cut to its training prefix, one
parameter vector per draw, and the held-out value of each target where one is
bound. For each draw it runs the bootstrap particle filter over the training
rows with the targets appended to their streams.

A target is appended as a row with no value, or with its held-out value wrapped
as [`ObsCell::HeldOut`](rust:sim::inference::multi_stream_obs::ObsCell). The
filter assimilates an `ObsCell::Scalar` and nothing else. At a stop where no
stream holds a `Scalar` it reweights nothing and does not resample. The driver
has no path that makes a `Scalar` for a target, and
[`declared_read`](rust:cli::fit::forecast_driver::declared_read) refuses to read
a prediction at a stop whose cell was assimilated, or at a time that ends no
declared period of the stream.

After the origin, therefore, every particle is propagated and none is reweighted
or copied. At each target's end the driver reads, per particle, a draw of the
observation from the stream's observation model. A forecast is a draw of the
observed quantity: it includes process noise, the declared observation noise,
and parameter uncertainty when there is more than one draw. Because no particle
is copied after the origin, particle _i_ of draw _m_ is one trajectory across
every target, and its draws at successive targets form one joint sample.
`forecast-samples.tsv` holds a subset of these.

Draws are pooled with equal weights. The forecast approximates the integral over
θ of p(y_future | θ, training rows) · p(θ | training rows), with the state
conditioned on the training rows. No draw is dropped. A draw whose filter
collapses, or a particle that dies after the origin from a numerical failure of
the step, fails the forecast with the draw, the time and the cause. A dropped
draw would bias the pool toward the parameter values at which the filter
survives.

The driver runs a fresh filter per draw whatever method produced the draws. A
PGAS fit also stores sampled state paths, and the forecast does not read them.

### 3.6 What is not a type

There is no wrapper type per role of a time (a label, a period's end, an
origin). Inside the engines every time is an `f64` in model time
([Dates and calendar time](../dates.md), "The one rule"). The confusion between
a label and an end is handled by `Measures`, which gives the end a name, and not
by distinct numeric types.

A row has no identity beyond its stream, its position and the digest of the
prefix it belongs to. Every set of rows this page needs is a prefix of a
stream's rows.

## 4. Scoring

### 4.1 Proper scoring rules

A probabilistic forecast is a distribution, and a scoring rule assigns a number
to a distribution and the value that was observed. A rule is _proper_ if a
forecaster's expected score is best when the distribution reported is the one
the forecaster believes, and _strictly proper_ if that report is the only best
one (Gneiting and Raftery 2007). A rule that is not proper can be improved by
reporting something other than the model's actual predictive distribution, so it
cannot be used to compare models.

camdl's primary rule is the logarithmic score, the log of the predictive density
at the observed value, which is strictly proper (Gneiting and Raftery 2007).
Scoring each observation by the predictive distribution formed before it was
seen is the prequential approach of Dawid (1984). `camdl score` also writes the
CRPS, the weighted interval score of Bracher et al. (2021), a randomized PIT and
interval coverage. Bracher et al. note that the log score cannot be computed
from a forecast stored as quantiles; camdl computes it because it has the model
that made the forecast. The definitions, the estimators and how to read each
column are in [Model comparison](../methods/model-comparison.md), §4, and the
columns of the score files are in the run specification, §2.2.4.

### 4.2 The log score of a forecast target

The predictive distribution of a fixed-origin forecast at a target is the
equal-weight mixture over _M_ draws and _N_ particles per draw (§3.5). Its
density at an observed value `y` is estimated by the mean, over the `M·N`
samples, of the observation density of `y` at each particle's state:

```text
p̂(y) = (1 / MN) · Σ_{m,i} p(y | x_{m,i}, θ_m)
log p̂(y) = logsumexp_{m,i} ℓ_{m,i} − log(MN),   ℓ_{m,i} = log p(y | x_{m,i}, θ_m)
```

The per-particle log densities `ℓ` come from the filter. When a target carries a
held-out value, the filter evaluates the stream's observation density of that
value at each particle at the target's end. This uses the likelihood code a fit
uses, at a stop where nothing has been assimilated, and the result enters no
weight. The driver returns the values as
[`Forecast::held_out_log_liks`](rust:cli::fit::forecast_driver::Forecast), and
[`log_mean_exp`](rust:cli::fit::forecast_score::log_mean_exp) reduces them.

The score is the log of the mean density. The mean of per-draw log scores is a
different quantity. By Jensen's inequality it is no larger, and it measures the
average quality of the single-draw forecasts, where the forecast camdl writes is
the pooled one. A `NaN` among the terms is refused, so a missing term cannot be
read as a score.

Two log scores are written and labelled apart. The _marginal_ score of one
target is the formula above. The _joint_ score of several targets sums each
sample's log densities over the targets before the log-sum-exp. Because sample
(_m_, _i_) is one trajectory across every target, this is the log predictive
density of the targets' values together. It is not the sum of the marginal
scores unless the targets are independent under the forecast.

The log score is an estimate from a finite number of particles. The mean density
is unbiased and its log is biased downward.
[Model comparison](../methods/model-comparison.md), §4.4, covers what that does
to a difference between two models.

A held-out value can be left unscored without refusing the forecast
([`HeldOutValues`](rust:cli::fit::forecast_driver::HeldOutValues)): when the row
is `NA`, when the observation model's data checks refuse the value against the
target's design, such as a count above its number of trials, and for every
design-reading stream of a prospective forecast, whose held-out value was
realised under a design other than the one supplied.

### 4.3 The outcome file

`camdl predict` computes a forecast without any held-out value. A stored fit has
none, and held-out outcomes are not part of a prediction's identity.
`camdl score` takes a stored prediction and one outcome file per stream and
stores the score inside the prediction. A forecast made today and scored next
month is one stored prediction with one score.

An outcome file given with `--data` is the whole series: the rows the prediction
conditioned on, then the rows after them. Before anything is computed,
[`check_agreement`](rust:cli::score_command::check_agreement) reads the file
with the loader the prediction used and compares it with the stored conditioning
rows, row by row, by row digest. It refuses a file that lacks one of them or
differs in one, naming the stream, the row and both values. Without this check a
user could score a forecast against a file from another source, or a revised
series, whose later rows do not continue the series the state was conditioned
on. Only the rows after the conditioning rows
([`OutcomeBinding::after_conditioning`](rust:cli::fit::scoring::OutcomeBinding))
can supply an outcome.

`--data-held-out` gives the later rows alone. Each must end after the stream's
last conditioning row, and the command joins them to the stored conditioning
rows, so both spellings of one outcome give one score.

The stored files of a forecast hold summaries and five particles per draw, which
is not enough to score. The score therefore computes the prediction again, with
each draw's own seed and the outcomes bound as held-out cells. A bound outcome
changes no weight and draws no random number, so the recomputed prediction must
equal the stored one. `check_recomputed` compares the run id and every file byte
for byte and refuses to write a score on any difference. This is the check that
a score did not change the forecast it scores.

For a forecast, each target is scored against the one outcome row that measures
the same period, with ends compared within `1e-9`. Two rows that measure one
target are refused. A target with no such row is not scored and is named in the
score's notes.

### 4.4 Which rows a held-out score counts

Every scored row carries its `role` (§1.4), read from the row's position against
its stream's training prefix. A forecast target's role is `future`. The summary
table never pools rows of different roles, modes, streams or leads, so a
held-out mean score is a mean over `held_out` rows of one stream in one mode.

The score as a whole carries a `conditioning` label
([`Conditioning`](rust:score::trace::Conditioning)), which says what the
parameters were fitted to. It is read from the training rows recorded in the
stored prediction's manifest. Before scoring, `camdl score` refuses a stored
prediction whose `predictive*.json` or `forecast.json` is not current, by its
schema tag, or does not record its training rows; without that refusal, a
held-out score of such a prediction would be labelled `in_sample`. The labels
are:

- `hold_out_tail` when the parameters come from a stored fit, and either the
  fit's recorded partition held rows out or the scored rows end after its
  training extent. The label rests on the recorded partition, whose digests were
  checked when the prediction was made.
- `in_sample` for a stored fit otherwise.
- `hold_out_declared` for a parameter point given with `--holdout-after`. No fit
  record exists, so the claim is not verified.
- `unknown` for a parameter point given without it.

The label is hold-out _tail_ because camdl's holdout is always the end of the
series (§1.1). Under `holdout_after` every row that did not train ends after the
cutoff. Under `[data.holdout]` the fit refuses a holdout row that ends at or
before the training extent, where the extent is taken over every bound stream.
Without that refusal, a stream with no holdout file could train past another
stream's held-out rows, and those rows would be scored as held out although the
parameters had seen data from the same days.

A held-out row of one stream can cover days on which another stream has training
rows. In the two-stream example, the `weekly` row labelled 1 February covers 26
January to 1 February, and the `daily` rows for 26 to 28 January were used for
training. The row is still held out: its own value was never fitted, and a
forecast made at the origin would have had those daily rows.

In the one-step mode, held-out rows are scored with assimilation. The filter
predicts the row labelled 22 February, is scored on it, assimilates it, and then
predicts the row labelled 1 March. The parameters stay as fitted to rows 1 to 6.
This is a sequence of one-week forecasts with fixed parameters, and its scores
are not comparable with the forecast mode's scores at leads of 14, 21 and 28
days. `camdl compare` refuses to difference rows of different modes or leads
([Model comparison](../methods/model-comparison.md), §2).

## 5. What is not supported

Each of these is refused or absent.

- A rolling origin. One fit has one training extent and a forecast starts there.
  Conditioning the state on rows after the fit with the parameters left as
  fitted, which would make the conditioning rows a larger set than the training
  rows, is not implemented in the forecast mode.
- A forecast origin chosen by the user. `ForecastPlan::new` has no origin
  argument.
- A fixed-origin forecast on the ODE backend. The value of a flow target would
  be the difference of the cumulative flow at the target's two ends on each
  draw's trajectory, plus observation noise. It is refused, and an ODE fit is
  predicted with `--mode free_forward`.
- A calendar period as a uniform `covers` form. A stream of calendar months or
  epidemiological weeks uses window columns, and needs an `emit_schedule` or
  held-out rows to have targets (§1.5).
- A target aggregated from a finer stream, such as a weekly target from a daily
  stream.
- Data vintages. The data are taken as given. A value revised after the forecast
  origin is not distinguished from the value known at the origin, except that an
  edit to a training row is refused by the digest.
- Forecast hub formats. camdl does not read or write a hub's files (§1.2).

## 6. Where to read the code

| File, under `rust/crates/cli/src/` | Holds                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| `fit/observed_rows.rs`             | the row reader, `ObservedRow` and `Measures`                                         |
| `fit/training_partition.rs`        | `TrainingPartition`, the cutoff rule, the row digest, `observed_window`              |
| `fit/forecast_plan.rs`             | `ForecastPlan`, `Target`, the three target sources and the refusals                  |
| `fit/forecast_driver.rs`           | `fixed_origin_forecast`, `StateConditioning`, `HeldOutValues`                        |
| `fit/forecast_covariates.rs`       | which forcings a forecast reads after its origin, and the `--covariate` statements   |
| `fit/forecast_output.rs`           | `camdl predict --mode forecast`: the refusals, the plan, the tables, `forecast.json` |
| `fit/forecast_score.rs`            | the marginal and joint log scores                                                    |
| `fit/scoring.rs`                   | the rules applied to each scored row, and `OutcomeBinding`                           |
| `score_command.rs`                 | `camdl score`: the outcome files, the agreement check, the recomputation             |
| `fit/predict.rs`                   | the free-forward and one-step modes, `RowRole`, the prediction's options             |

The types that label a score (`Conditioning`, `Provenance`, `PredictiveBasis`)
are in `rust/crates/score/src/trace.rs`. `ObsCell` and the filter's treatment of
a held-out cell are in `rust/crates/sim/src/inference/multi_stream_obs.rs` and
`particle_filter.rs`.

The tests that hold the properties on this page are
`rust/crates/cli/src/fit/forecast_driver_tests.rs` (the week under way, joint
samples, that held-out values change only the held-out log-likelihoods) and,
under `rust/crates/cli/tests/`, `training_partition.rs`,
`fit_predict_forecast.rs`, `gh1079_one_step_row_roles.rs`, `holdout_applied.rs`
and `forcing_out_of_range.rs`.

## References

- Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating
  epidemic forecasts in an interval format. _PLOS Computational Biology_, 17(2),
  e1008618. <https://doi.org/10.1371/journal.pcbi.1008618>
- CDC COVID-19 Forecast Hub. `README.md` and `model-output/README.md`.
  <https://github.com/CDCgov/covid19-forecast-hub>
- Dawid, A. P. (1984). Statistical theory: the prequential approach. _Journal of
  the Royal Statistical Society, Series A_, 147(2), 278–292.
  <https://doi.org/10.2307/2981683>
- FluSight Forecast Hub. `model-output/README.md`.
  <https://github.com/cdcepi/FluSight-forecast-hub>
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. _Journal of the American Statistical Association_,
  102(477), 359–378. <https://doi.org/10.1198/016214506000001437>
- Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and
  Practice_ (3rd ed.). OTexts. <https://otexts.com/fpp3/>. §5.8, "Evaluating
  point forecast accuracy"; §5.10, "Time series cross-validation"; §7.6,
  "Forecasting with regression".
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in
  machine-learning-based science. _Patterns_, 4(9), 100804.
  <https://doi.org/10.1016/j.patter.2023.100804>
- Kaufman, S., Rosset, S., & Perlich, C. (2011). Leakage in data mining:
  formulation, detection, and avoidance. _Proceedings of the 17th ACM SIGKDD
  International Conference on Knowledge Discovery and Data Mining_, 556–563.
  <https://doi.org/10.1145/2020408.2020496>
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an
  analysis and review. _International Journal of Forecasting_, 16(4), 437–450.
  <https://doi.org/10.1016/S0169-2070(00)00065-0>
- US COVID-19 Forecast Hub. `data-processed/README.md`.
  <https://github.com/reichlab/covid19-forecast-hub>
