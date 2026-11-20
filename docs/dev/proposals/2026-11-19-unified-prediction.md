# Proposal: one predictive command, one prediction table, one score command

**Status:** Draft for the maintainer's review. Not approved; nothing here is to
be implemented until it is. **Date:** 2026-11-20. **Depends on:**
`docs/dev/temporal-architecture.md` (the temporal spec) and
`2026-11-15-temporal-architecture-migration.md` (its runbook), whose Stage 5 is
on `main`. **Amends:** `2026-11-14-score-artifact.md` (the sections listed under
[What changes in the score-artifact proposal](#what-changes-in-the-score-artifact-proposal))
and the runbook's Stage 6. **Supersedes:**
`2026-08-29-honest-predictive-evaluation.md` §3.7.2, §3.8 and its Stage 5.
**Decisions it assumes:** D100 to D105 and D108 in the maintainer's docket, each
marked where it is used; the draft takes the recommended option of each.

Code references are to `main` at `54eb15405` and name the function or type, not
a line. A statement marked _run_ was checked by running a release binary (built
at `c92e7c9e3`); one marked _read_ was read in the code and not run.

## Summary

camdl answers "what does this model predict?" in four commands, with four table
layouts and three different rules for what a prediction was allowed to see. This
proposal gives prediction one command, `camdl predict`, one typed description of
a prediction and one table. Scoring is a second command, `camdl score`, the only
one that reads the outcomes a prediction did not see. `pfilter` keeps the
likelihood at one parameter vector, and `simulate` stays a generator that reads
no data.

Five capabilities are new:

- a prediction from a fixed parameter vector, with no fit;
- one-step-ahead prediction through held-out rows, every row marked;
- forecasts from many forecast origins, scored by lead time;
- one plot-ready table over training and held-out rows;
- a stated treatment of every covariate a forecast reads after its origin, with
  no silent default.

## The problem, from the start

### One object

A partially observed model has parameters `θ`, a latent state `x(t)` and
observed rows `y`. Every question this proposal is about has the same form:

> the distribution of a row `y`, given parameters that were fitted to some rows,
> and a state that has seen some rows.

Two sets of rows therefore describe any prediction, and they are different sets:

- the **parameter set**, the rows `θ` was fitted to;
- the **state set**, the rows the state `x` had been conditioned on (by a
  particle filter) when the row was predicted.

Everything else is a consequence. A row is _held out_ when it is outside the
parameter set. A prediction's _lead time_ is how far the predicted row ends
after the last row in the state set. Simulation, one-step-ahead prediction,
forecasting and free-running projection are four choices of state set, and
"in-sample" against "held-out" is a statement about the parameter set. The
temporal spec already says this (§2.4, and §2.6: "a free-running projection from
the start of the run is a fixed-origin forecast whose origin is the run's start
and whose state set is empty, not a fourth mode"); the code and the command line
do not yet follow it.

### The modes, as choices of state set

Take one weekly stream with twelve rows, the first seven fitted to:

```
row ends      1   2   3   4   5   6   7   8   9   10  11  12
            ──┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴──▶ time
              └──── parameter set ────┘   └──── held out ───┘
                                      ▲
                               training extent
```

In the pictures below, `■` is a row the state has assimilated when the
prediction is made and `○` is a row being predicted.

```
simulate        ○   ○   ○   ○   ○   ○   ○   ○   ○   ○   ○   ○
                no data is read at all; the output is a draw from the model

free-forward    ○   ○   ○   ○   ○   ○   ○   ○   ○   ○   ○   ○
                the same state set (empty), at parameters fitted to rows 1-7,
                laid against the rows: does the fitted model generate data
                like these?

one-step        ■   ■   ■   ■   ○                                 predicting row 5
                ■   ■   ■   ■   ■   ■   ■   ■   ○                 predicting row 9
                each row is predicted from every earlier row and then
                assimilated; the state set grows by one row at a time, so the
                lead is always one period

forecast        ■   ■   ■   ■   ■   ■   ■ ┃ ○   ○   ○   ○         origin = 7, reach = 4 weeks
                the state set stops at the forecast origin; rows 8-11 are
                predicted with nothing assimilated in between, so their leads
                are 1, 2, 3 and 4 weeks
```

So a forecast differs from one-step prediction in one thing: after the origin,
nothing is assimilated. Three consequences follow from that and are what make it
a separate mode rather than a flag:

- **Lead time grows.** Row 11 is predicted four weeks ahead, row 8 one week
  ahead. One-step prediction only ever measures skill at one period's lead.
- **Targets come from a declaration, not from data.** One-step predicts the rows
  that exist. A forecast has to say which periods it predicts past the origin,
  where there may be no rows at all; they come from each stream's declared
  rhythm (spec §4.3) and are the plan's `Target`s.
- **One sample is one future.** With no resampling after the origin, particle
  `i` of draw `m` is one trajectory through every target, so its values at rows
  8 to 11 are a joint draw (D54, D72). One-step draws at different rows are not
  jointly distributed in any usable way.

One-step prediction is not "rolling origins with a reach of one period" computed
another way: it is that, and it is cheap, because a single filter pass yields
every row's one-period-ahead predictive before the row is assimilated. A
forecast of longer reach from each of many origins cannot be read off one pass.

### Many origins

A single origin gives one prediction at each lead, which is too few to say how
good the model is four weeks ahead. Moving the origin through the held-out rows
gives many. The parameters stay those of the one fit; only the state set grows
(D35: no refit). With `--origins "every 4 weeks from T to T+12 weeks"` and
`--horizon "8 weeks"` on a weekly stream:

```
weeks after T    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
               ──┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴─▶
origin T       ■ ┃ ○   ○   ○   ○   ○   ○   ○   ○
origin T+4     ■   ■   ■   ■   ■ ┃ ○   ○   ○   ○   ○   ○   ○   ○
origin T+8     ■   ■   ■   ■   ■   ■   ■   ■   ■ ┃ ○   ○   ○   ○   ○   ○   ○   ○
origin T+12    ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   ■ ┃ ○   ○   ○   ○   ○   ○   ○   ○

lead (weeks)                         1   2   3   4   5   6   7   8     (each origin's own targets)
```

Four origins, eight targets each: 32 rows of `predictions.tsv` for this stream,
keyed by (origin, target period). Each (origin, target) is one prediction at one
lead; the week ending six weeks after T is predicted twice, at leads of 6 weeks
(from T) and 2 weeks (from T+4). Scoring then groups by lead: the four
predictions at one week's lead give one row of the per-lead summary, the four at
two weeks' lead the next, and so on.

```
predictions.tsv (one stream, the observation quantity; other columns omitted)

origin   t_start  t_stop   horizon  lead  role    q05  q50  q95
T        T        T+7      0        7     future  ...
T        T+7      T+14     1        14    future
...
T+4w     T+28     T+35     0        7     future
...

summary.tsv, written by `camdl score` over those 32 scored targets

stream  mode      role      lead  n  log_score_mean  crps_mean  coverage_50  coverage_90
cases   forecast  held_out  7     4  ...
cases   forecast  held_out  14    4
...
cases   forecast  held_out  56    4
```

### What a band shows

For one row there are two different random quantities, and a band can be of
either:

- the **observation** `y`: a draw from the observation model. Its spread carries
  uncertainty about the parameters, about the state, and the observation noise
  (the negative-binomial or binomial scatter of a count around its mean). Data
  are compared with this one: about 90% of observed rows should fall in its 90%
  band, and every score is computed against it.
- the **expectation** `E[y | x, θ]`: the observation model's mean at one state
  and one parameter vector, with no noise drawn. It is not a single line. It is
  a distribution too, over particles (state uncertainty) and draws (parameter
  uncertainty), and its band is what is usually drawn as "the fitted curve with
  its uncertainty". Data are not expected to fall inside it.

The two have the same mean and different widths; by the law of total variance
the variance of the observation is the variance of the expectation plus the mean
of the observation model's own variance. The difference between the two bands is
therefore the part of the predictive spread that is observation noise, which is
the diagnostic use of asking for both. Today camdl writes the observation band
from `fit predict` and the expectation band from
`pfilter --save-filtered-streams`, and neither file says which it holds; a
reader who lays data over the second concludes the model is overconfident.

### Scoring

A prediction is made without the outcome. A **score** compares it with the
outcome once there is one: the log predictive density of the observed value (the
log score), the continuous ranked probability score, the probability integral
transform, and whether the value fell inside the 50% and 90% intervals. Three
reasons make it a separate object from the prediction:

- **The outcome must not be able to change the prediction.** A forecast made
  today and scored next month has to be the same file both times (D81). If
  observed values sat in the prediction's table, the prediction's content would
  depend on data from after its origin.
- **A score is only meaningful with its two sets attached.** The same number
  means different things when the parameters had seen the scored row (in-sample)
  and when they had not, and at one period's lead or eight. Stored with its
  parameter set, mode and lead, a score cannot be pooled or compared with one of
  a different kind by accident.
- **Scores are expensive and are compared across fits.** Today `compare`
  recomputes every score each time it runs and keeps none.

### Covariates

A forcing read from a data table (births, population, temperature) is an input
to the model at every time. After a forecast origin its values are of three
kinds that camdl cannot tell apart: fixed in advance or assumed as a scenario,
observed later, or missing. The first makes a forecast, the second an after-the-
fact evaluation that flatters a model's skill, the third has no answer. This
proposal makes the user state which, per forcing.

## Terms

Terms from the temporal spec are used as defined there (§2.2 to §2.6, §5.5).

- **Stream.** One concrete observation series of a model's `observations {}`
  block.
- **Period, end.** What a data row measures (a count over a stretch of time) and
  the instant that stretch ends, which is when the row is scored.
- **Instant, duration.** An instant is a point on the model's time axis, written
  on the command line as a date (for a model with a calendar origin) or a number
  in the model's time unit. A duration is a fixed length of time, written
  `"4 weeks"` or `"28 days"` or a bare number in the model's time unit, and
  parsed by the existing `forecast_plan::parse_reach`; months and years are
  refused under a calendar origin because they are not a fixed length (decision
  F). Two durations appear: a forecast's **reach**, how far past each origin its
  targets extend, and the **step** between successive origins.
- **Training rows, training extent.** The rows a fit's parameters were fitted
  to, and the latest end among them.
- **Parameter set, state set.** As above.
- **Forecast origin.** The instant after which a forecast assimilates no row.
- **Target.** One stream's period that a forecast predicts.
- **Lead time.** The duration from the origin to the target's end.
- **Horizon.** A target's position among its stream's targets, counting the
  first one that ends after the origin as 0 (D41). The command-line flag
  `--horizon "4 weeks"` is the reach, a duration; the two share a word for
  historical reasons and D41 fixed which is which.
- **Fixed point.** One parameter vector supplied by the user or taken from a
  single-estimate fit, as opposed to draws from a posterior.

## What exists today

### The commands

| Command                                       | Parameters                       | State set                                               | Output                                                                                   |
| --------------------------------------------- | -------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `simulate`                                    | one vector, or draws (`--draws`) | none                                                    | trajectories; `--obs` files                                                              |
| `simulate --init-state FILE`                  | one vector                       | whatever the filter that wrote `FILE` saw               | the same, from the saved time on                                                         |
| `simulate --init-state fit --draws posterior` | a PGAS fit's draws               | each draw's own sampled path                            | the same                                                                                 |
| `pfilter --save-prequential`                  | one vector                       | per row, every earlier row                              | `STEM.streams.tsv`: observed value, observation band, log score, CRPS, PIT               |
| `pfilter --save-filtered-streams`             | one vector                       | per row, every earlier row; and with the row itself too | expectation band, before (`predictive`) and after (`filtered`) the row is weighted       |
| `fit predict --mode free_forward`             | posterior draws                  | none                                                    | `predictive/<stream>.tsv`: observation band, plus a tail past the data                   |
| `fit predict --mode one_step`                 | posterior draws                  | per row, every earlier row                              | the same file, more rows                                                                 |
| `fit predict --mode forecast --horizon D`     | posterior draws                  | the training rows                                       | `forecast/<stream>.tsv`, `forecast-samples.tsv`, `forecast.json`                         |
| `compare`                                     | a fit's draws                    | per scored row, every earlier row                       | one-step scores, derived by child `pfilter` processes in a temporary directory, not kept |

### What the code already has

The pieces are further along than the command line shows.

- **The plan.** `fit/forecast_plan.rs`: `ForecastPlan` holds an origin, a reach,
  the fit's `TrainingPartition`, and the `Target`s (stream, period, horizon,
  lead, the held-out row that measures the same period if one exists, and the
  design values the stream's likelihood reads there). `ForecastPlan::new` builds
  it before any filtering and hard-codes `origin = parameters.extent()`.
- **The driver.** `fit/forecast_driver.rs`, `fixed_origin_forecast`: for each
  posterior draw it runs the existing bootstrap particle filter over the
  training rows plus the targets, each target appended to its stream as a
  declared period with no value the filter weights on. It reads the predictive
  draws at each target's end. A target whose held-out value exists carries it as
  a held-out cell (D79): the filter evaluates its likelihood per particle and
  never weights on it. It returns a `Forecast` with `draws`,
  `held_out_log_liks`, an optional `expected` (each particle's `E[y | x, θ]`,
  which no command asks for), and a `StateConditioning` record of the row
  prefixes that were actually handed to the filter.
- **The log scores.** `fit/forecast_score.rs`: the marginal log score of a
  target, `logsumexp − log(MN)` over `N` particles of `M` draws, and the joint
  log score of several. No caller outside tests; nothing writes them.
- **The one-step band.** `fit/predict.rs`, `one_step_bands`: one
  `bootstrap_filter` pass per draw over every bound row, pooling each row's
  predictive draws before the row is weighted.
- **The score types.** The `score` crate has `PrequentialTrace` (per row and
  stream: log score, CRPS, randomized PIT, interval), and three labels:
  `Provenance` (a plug-in point or a mixture over `n_draws` posterior draws),
  `Conditioning` (what the parameters saw: `InSample`, or `HoldOutTail` with a
  `train_end`), and `PredictiveBasis` (a particle filter or an ODE trajectory).
  `pfilter --save-prequential` writes one; `compare` derives one per fit and
  discards it.
- **Row marking.** `fit/predict.rs` has `RowRole` (`Trained`, `HeldOut`,
  `Future`) and a `ParameterSet` struct that reads the fit's recorded partition
  and verifies it against the bound files (gh#1079, `7918f460a..ec75f4c66`).

### What it costs a user

- **A fixed parameter vector cannot be predicted from.** `fit predict --params`
  is refused by the argument parser (_run_), and a single-estimate fit (IF2, the
  NLopt optimizers) is refused by decision D51 (_read_). The one-step predictive
  at a fixed vector exists only under `pfilter`, and a forecast from one only
  through `simulate --init-state`, which has no targets, lead times or record of
  what the state saw.
- **The same quantity has two names and two layouts.** `pfilter`'s
  `STEM.streams.tsv` has `t stream y_obs y_pred_q05 … log_score crps pit`; its
  filtered-streams file has `time stream phase mean q05 … q95` (_run_).
  `fit predict` writes
  `scenario time <dims> horizon role treatment … q05 …
  q95`, where the
  `horizon` column holds the mode name, while the forecast table's `horizon`
  column holds an integer (_read_; gh#1302).
- **No table states which quantity its band shows.**
- **Held-out rows are marked in one file only.** `predictive/<stream>.tsv` has
  the `role` column; `pfilter --score-from 21` drops the rows before day 21
  rather than marking them (_run_), and stamps a fixed-point trace `in_sample`
  (gh#1305). A fit with separate holdout files gets no one-step row for them
  (gh#1309).
- **There are two forecasts in one command.** `fit predict` with no flags writes
  a free-forward tail past the last row, on a grid built from the most common
  gap between labels, starting at the largest label of the whole file including
  held-out rows (`forecast_times`, _read_; gh#1151, gh#1152). The runbook's
  Stage 5 lists this code for deletion; it is still there.
- **Forecast scores are computed and thrown away,** and one-step scores are
  recomputed by every `compare`.
- **Only one forecast origin exists,** the training extent (D57).
- **Two rules for a failed draw.** The one-step band skips a draw whose filter
  degenerates and counts it; the forecast fails outright (D53). Neither table
  says how many draws stand behind a row.

### Covariates today

A forcing is one of the kinds of `docs/dev/forcing-kinds.md`. How each behaves
when a run goes past the fit's data:

| Kind                                                   | Past its data                                                                                            |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `sinusoidal`, `fourier`, `periodic`, `periodic_spline` | defined for all time; repeats                                                                            |
| a formula in `t`                                       | defined for all time                                                                                     |
| `piecewise`                                            | holds its last value for all time, with no message                                                       |
| `interpolated` (linear, constant, cubic spline)        | an error past its last row, naming the forcing and its range in model time (D44, D48; no dates, gh#1304) |
| `bspline`                                              | an error outside its window                                                                              |

Three things follow for a forecast (_read_; none run).

- **The only way out of the error is all or nothing.** The run-level switch
  `--allow-forcing-extrapolation` (in a fit config,
  `[config] allow_forcing_extrapolation = true`) lets every ranged forcing in
  the model hold its end value. `fit predict` carries the fit's setting forward.
  gh#1300 proposes a per-forcing declaration in the model instead.
- **A forecast cannot be given covariate values the fit did not have.** A fit
  binds each `external("NAME")` table once, folds its bytes into the fit's
  identity and keeps a copy; every later reader of the fit binds that copy (run
  spec, "A stored table is checked before it is bound"). `fit predict` has no
  `--table`. So the values a forecast reads after its origin have to be in the
  table at fit time, and extending the table means refitting.
- **Values after the origin are read without a word.** If the table has rows
  after the origin, the forecast uses them, and nothing records that it did
  (gh#1083; the spec's §4.6 note is not built).

The error is also raised late. The range check reads the values the run actually
evaluated, so a forecast runs its filter through every training row and fails
when the first particle steps past the table.

## How others divide the same work

These were read in their current documentation.

- **pomp** (King, Nguyen and Ionides 2016; manual 6.5) keeps three functions.
  `simulate` draws from the model with no conditioning on data. `pfilter` runs
  at one parameter vector and returns, per observation time, the conditional log
  likelihood (`cond_logLik`), the effective sample size, and optionally the
  prediction and filtering means of the state (`pred.mean`, `filter.mean`).
  `forecast()` returns the observation mean at the mean predicted state, one
  number per time and not a distribution. Forecasting past the data is user
  code: the Ebola lesson of the SBIED course saves the final particles, writes
  them into initial-value parameters, and calls `simulate`. pomp has no object
  for a forecast, no origin or lead time, and no marking of rows as training or
  held-out.
- **The Hubverse** (hubverse.io documentation, schema 6.0) defines a forecast as
  rows of one table: task-identifying columns (`origin_date`, `horizon`,
  `target`, `location`, the derived `target_end_date`), then `output_type`
  (`mean`, `median`, `quantile`, `cdf`, `pmf`, `sample`), `output_type_id` and
  `value`. Observations live elsewhere (`target-data/`) and are joined only when
  scoring. Samples sharing an `output_type_id` are one joint draw across
  horizons.
- **scoringutils** (Bosse et al.; manual 2.3) scores a table that holds
  `observed` and `predicted` side by side, keyed by a _forecast unit_, and
  summarises with `summarise_scores(by = …)`. Its defaults for quantile
  forecasts include 50% and 90% interval coverage. Its sample-based log score is
  a kernel density estimate that its manual says is for continuous predictions
  only.

What is taken, and what is not:

- **Taken from the Hubverse:** the key of a prediction row (origin, target
  period, horizon) and the separation of observations from model output. Both
  are forced by the problem and not conventions: they are the state set and D81.
- **Taken from scoringutils:** a score table that holds observed and predicted
  together and can be summarised by any grouping, with coverage at 50% and 90%.
- **Not taken from either:** the log score from samples or quantiles. Both
  systems receive forecasts as samples or quantiles from models they cannot see
  inside, so a density has to be estimated. camdl has the model. A particle
  filter gives the predictive density of the observed value directly, as the
  average over particles and draws of the observation model's density, with no
  kernel and no binning (Jordan, Krüger and Lerch 2019, §2.2, recommend exactly
  this mixture when it is available). It is exact for counts, where a kernel
  estimate is not defined.
- **Not taken from pomp:** forecasting by restarting `simulate` from saved
  particles. A saved state restores compartment counts and nothing else, so it
  cannot produce the row of a reporting period under way at the restart, and it
  keeps no record of what the state saw. A forecast here never leaves the
  filter: the targets are appended to the filter's own schedule, so the
  accumulators, the random streams and the record of assimilated rows are the
  filter's.
- **Neither system has the parameter set.** A hub scores whatever is submitted;
  whether a model's parameters had seen the scored weeks is outside its table.
  For a model comparison that is the first thing to get wrong, so here it is a
  column.

For time-ordered data the sum of one-step log predictive densities is the
prequential log likelihood (Dawid 1984, cited through Gneiting and Raftery 2007,
§7.1). Hyndman and Athanasopoulos (_Forecasting: Principles and Practice_, 3rd
ed., §5.8 and §5.10) call errors on training rows residuals and not forecast
errors, and call scoring from many origins evaluation on a rolling forecasting
origin; the names here follow them.

## The design

### Division of labour

| Command    | What it is for                                                                  | Reads observed data                         |
| ---------- | ------------------------------------------------------------------------------- | ------------------------------------------- |
| `simulate` | generate from the model                                                         | never                                       |
| `pfilter`  | the likelihood at one parameter vector, and diagnostics of that filter run      | every bound row                             |
| `predict`  | predictions: bands and samples, each with its two sets                          | only the rows in the prediction's state set |
| `score`    | scores of a prediction against outcomes; the plot-ready table with observations | the outcomes of every scored row            |
| `compare`  | differences between stored scores of several fits                               | nothing: it reads score artifacts           |

`camdl predict` replaces `camdl fit predict` (D100): a prediction from a fixed
point has no fit, so the command cannot sit under `fit`. `simulate --init-state`
remains as a lower-level way to continue a simulation from saved counts (D64);
its documentation stops calling it forecasting.

### What is removed from existing commands

Nothing a user can compute today becomes impossible, with one exception, the
first item.

| Today                                                                                  | Under this proposal                                                                                                                     | Stage |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `fit predict` writes a free-forward tail past the last row                             | removed and not replaced: its periods were inferred from gaps between rows (gh#1151, gh#1152). `--mode forecast` predicts past the data | 1     |
| `camdl fit predict`                                                                    | `camdl predict`; the old spelling fails naming the new one                                                                              | 3     |
| `predictive/<stream>.tsv`, `forecast/<stream>.tsv`, `predictive.json`, `forecast.json` | `predictions.tsv` and `prediction.json`; `camdl show` and `camdl cat` read the new files                                                | 2     |
| convergence columns (`fit_rhat_max`, …) in every predictive row                        | a sibling `convergence.tsv`                                                                                                             | 2     |
| the `horizon` column holding a mode name                                               | a `mode` column; `horizon` is always an integer                                                                                         | 1     |
| `fit predict` carries the fit's `allow_forcing_extrapolation` into a forecast          | a per-forcing statement (below); the run-level switch is not consulted after a forecast origin                                          | 4     |
| `compare` derives one-step scores itself and discards them                             | `compare` reads stored score artifacts, deriving a missing one through `score`; its command line is unchanged                           | 7     |
| `pfilter --save-prequential`, `--score-from`                                           | `camdl score MODEL --params FILE --mode one-step`, which marks rows instead of dropping them                                            | 8     |
| `pfilter --save-filtered-streams`, the `predictive` rows                               | `camdl predict --params FILE --mode one-step --quantity expectation`; the `filtered` rows stay in `pfilter` (D103)                      | 8     |
| a single-estimate fit is refused a prediction (D51)                                    | allowed and labelled (D101)                                                                                                             | 3     |

### Types

Each of the two sets is one type, the mode is one type, and what the filter
actually did is recorded separately from what was asked. Names that already
exist are kept.

#### Where the parameters came from

```rust
pub enum ParameterSource {
    /// Draws of a sampler fit (PMMH, PGAS, MH on ODE): a mixture over draws.
    Posterior { fit: FitRef, chains: ChainSelection, n_draws: usize },
    /// One parameter vector. The prediction carries no parameter uncertainty.
    FixedPoint { point: ParamPoint, from: PointOrigin },
}

pub enum PointOrigin {
    /// Given on the command line: `--params FILE` (repeatable, applied in
    /// order) and `--param NAME=VALUE` overrides, as `simulate` and `pfilter`
    /// take them. camdl knows nothing of how the values were obtained.
    Supplied { files: Vec<ParamsFile>, overrides: Vec<(String, f64)> },
    /// The single estimate of an IF2 or NLopt fit.
    FitEstimate { fit: FitRef },
}

/// One `--params` file: the path as given (for messages) and its digest.
pub struct ParamsFile { pub path: PathBuf, pub digest: ContentHash }
```

`ParamPoint` is the resolved vector, every model parameter with its value, and
is what enters the prediction's identity; `PointOrigin` is provenance for the
manifest. A supplied point is not one file: the existing flags allow several
files and single overrides, and the type records all of them.

`ParameterSource` replaces `ParamTreatment` (`fit/predict.rs`), whose `PlugIn`
variant exists only to be refused. It maps onto the label the `score` crate
already has, and that label is kept: `Posterior { n_draws }` to
`score::Provenance::Posterior { n_draws }`, `FixedPoint` to
`score::Provenance::PlugIn`. The table column `parameters` holds `posterior` or
`fixed_point`; a fixed-point manifest carries the sentence "fixed point: no
parameter uncertainty", and every message that prints a band from one repeats
it. Allowing it reverses D51 (D101).

#### The parameter set

```rust
pub enum ParameterSet {
    /// Read from a fit's record and verified against the data by row digest.
    Recorded(RecordedPartition),
    /// Stated by the user for a fixed point (`--trained-through INSTANT`).
    /// camdl cannot check it; every artifact says "declared, not verified".
    Declared(TrainingPartition),
    /// A fixed point with nothing stated.
    Unknown,
}
```

`RecordedPartition` is today's `ParameterSet` struct (`fit/predict.rs`: the
fit's record, the verified partition, the sources with a holdout file), renamed.
A fixed vector taken from a paper was fitted to some data, and camdl cannot know
which. Under `Unknown` every row is marked `unknown` and no score is stamped
held-out. Under `Declared` rows are marked from the stated cutoff with the rule
a fit uses (a row trains if its period ends at or before the cutoff, spec §4.1),
and a score from it is labelled "held-out as declared".

The `score` crate's `Conditioning` is the per-artifact form of the same fact and
is kept, with one variant added and one renamed in meaning:

| `ParameterSet`                         | `score::Conditioning`                 |
| -------------------------------------- | ------------------------------------- |
| `Recorded`, scored rows all outside it | `HoldOutTail { train_end }`           |
| `Recorded`, any scored row inside it   | `InSample`                            |
| `Declared`, scored rows all outside it | `HoldOutDeclared { train_end }` (new) |
| `Unknown`                              | `Unknown` (new)                       |

Today a fixed-point trace is stamped `InSample` because that is the enum's
default (gh#1305); `Unknown` is what it should say.

#### The mode: which state set each predicted row has

```rust
pub enum PredictionMode {
    /// Empty state set: the model run forward from its start.
    FreeForward,
    /// Each row's state set is every row, of every stream, ending before it.
    OneStep,
    /// At each origin the state set is the rows ending at or before it.
    Forecast { origins: Origins, reach: Duration },
}

pub enum Origins {
    /// The training extent. Today's `--mode forecast`.
    TrainingExtent,
    /// Listed instants (`--origin INSTANT`, repeatable).
    At(Vec<Instant>),
    /// `--origins "every STEP from FIRST to LAST"`: FIRST, FIRST + STEP, …,
    /// up to and including LAST when it is on the sequence.
    Every { step: Duration, from: Instant, to: Instant },
}
```

`PredictionMode` replaces the `Horizon` enum (`fit/predict.rs`) and
`PredictModeArg`, and the separate entry into forecast mode. It is not named
`Conditioning` because the `score` crate has that name for the parameter set.

Rules for origins, each a refusal at plan time with the instant named:

- an origin before the first row of every stream has an empty state set, which
  is free-forward, and is refused with that suggestion;
- an origin is used exactly as given and never moved to a period boundary. When
  it falls inside a stream's period, that period is the stream's horizon-0
  target and its lead is shorter than a full period (spec §2.5); a note says so
  once per stream, because such targets land in a lead group of their own;
- with a fixed point whose parameter set is `Unknown`, `TrainingExtent` has no
  meaning and an origin must be given;
- an origin inside the training window is allowed. Its targets are rows the
  parameters were fitted to, so they are marked `trained`: useful for seeing how
  the model tracks its own training series at four weeks' lead, and never pooled
  with held-out targets.

#### What the filter did

`PredictionMode` is the request. `StateConditioning` is the record, and it
already exists (`fit/forecast_driver.rs`): it is built only inside the driver,
from the rows actually handed to the filter, so no caller can claim a
conditioning that did not happen. It gains the two cases the other modes need.

```rust
pub enum StateConditioning {
    /// Free-forward: no row was handed to a filter.
    None,
    /// Forecast: per stream, the row prefix assimilated, with its digest,
    /// ending at `through` (the origin).
    Filtered { through: f64, rows: Vec<RowPrefix> },
    /// One-step: per stream, the rows filtered through; each row was
    /// predicted from the prefix ending before it.
    Sequential { rows: Vec<RowPrefix> },
}
```

It is needed for three things. The manifest states exactly which rows a
prediction read, by digest. The prediction's identity folds those digests, so
changing an assimilated row re-keys the prediction and changing a later one does
not (the no-leak test). And when an origin is later than the training extent the
state set is larger than the parameter set; the constructor checks that the two
agree on the rows they share, which is what stops a prediction being made with
parameters fitted to one version of a file and a state filtered through another.

#### The quantity

```rust
pub enum Quantity {
    /// A draw of y from the observation model.
    Observation,
    /// E[y | x, θ] per particle and draw: no observation noise.
    Expectation,
}
```

`--quantity observation|expectation|both`; the default is `observation`, the
quantity data are compared with and scores are computed against. The column is
always written, so a band cannot be mistaken for the other.

#### The row

```rust
pub enum RowRole { Trained, HeldOut, Future, Unknown }

pub struct PredictionRow {
    pub scenario: String,
    pub stream: String,
    pub stratum: Vec<(String, String)>,
    pub measures: Measures,          // the period or instant predicted
    pub mode: ModeKind,              // free_forward | one_step | forecast
    pub origin: f64,
    pub horizon: Option<i64>,        // forecast only
    pub lead: f64,                   // measures.end() − origin
    pub role: RowRole,
    pub quantity: Quantity,
    pub parameters: ParameterKind,   // posterior | fixed_point
    pub mean: f64,
    pub quantiles: Quantiles,        // the existing fixed set, q05 … q95
    pub n_draws: usize,              // parameter draws pooled
    pub n_samples: usize,            // draws × particles pooled
}
```

`origin` is defined in every mode, so `lead` is too: the run's start under
free-forward; under one-step, the end of the latest row of any stream
assimilated before this row, which varies by row (spec §2.6); the forecast
origin under forecast. `n_draws` makes the one-step rule visible: a draw whose
filter degenerated is left out of that mode's rows, and the count shows it.

`role` describes the row against the parameter set, using only rows the
prediction itself read:

- `trained`: the row is in the parameter set.
- `held_out`: a data row outside the parameter set that this prediction
  assimilated (a one-step row, or a row before a later forecast origin).
- `future`: the prediction read no data row for this period. A forecast target
  is `future` in the prediction even when a held-out row for it exists, because
  the prediction never read that row (D81). The score table, which does read it,
  marks it `held_out`.
- `unknown`: the parameter set is `Unknown`.

### How a prediction is computed

One planner and one driver serve the three modes; today there are three code
paths.

**Free-forward.** Unchanged: the simulation job replays each draw from the run's
start and draws observations at each bound row. No filter runs. The tail past
the data is deleted (stage 1).

**One-step.** One bootstrap-filter pass per parameter draw over the rows of the
state set, as `one_step_bands` does today. Two changes. The rows are read by the
reader the forecast plan uses, so rows of a `[data.holdout]` file are filtered
through and predicted like rows past `holdout_after` (gh#1309). And when
`camdl score` runs it, the filter's prequential record supplies each row's
per-particle likelihood, so the row's log score needs no second pass.

**Forecast.** `ForecastPlan::new` gains the origin and the state set as
arguments in place of deriving both from the partition:

```rust
impl ForecastPlan {
    pub(crate) fn new(
        parameters: &ParameterSet,
        origin: f64,
        reach: f64,
        streams: &[PlanStream],   // per stream: the rows ending at or before
                                  // the origin, and the rows after it
    ) -> Result<Self, PlanError>;
}
```

`PlanStream.training` becomes `assimilated`, the stream's rows ending at or
before the origin whether or not they trained; `held_out` becomes `later`, the
rows after it, which only name a target's `observed` row as now. Everything else
in the plan is unchanged: `stream_targets` already takes the origin as an
argument. The driver changes one line of meaning, handing the filter each
stream's `assimilated` rows where it now hands the training prefix.

A run with several origins builds one plan per origin and calls the driver once
per plan. Three properties of that choice:

- **It is exact and needs no change to the filter.** `particle_filter.rs` is a
  high-risk file and is not touched for origins.
- **Its cost is known.** Each origin repeats the filter over every row before
  it, so the work is (origins) × (draws) × (particles) × (steps before the
  origin). The stage that builds it reports the measured wall time for a
  four-year weekly rolling forecast on the He et al. fixture
  (`tests/external/cases/he2010_forward`) in its PR.
- **Origins share random numbers.** A draw's seed is `draw_seed(base_seed, d)`
  for every origin, so two origins' filters are bit-identical up to the earlier
  origin. A multi-origin run therefore equals the single-origin runs row for
  row, and differences between adjacent origins' forecasts are not inflated by
  independent Monte Carlo noise.

Filtering once and copying the particles at each origin would cost one pass. It
needs a complete snapshot of a particle (counts, the running totals of every
open reporting period, and the random stream), which is the same missing
primitive that makes `simulate --init-state` inexact inside a period (PR #1158,
D64). It is left as a later, measured optimisation that would serve both (D102).

**The expectation.** The observation draws come from the filter's prequential
record, which holds each stream's per-particle predictive draw at selected stops
(`PrequentialStops`). The expectation needs the same record to hold each
stream's per-particle mean, `obs_model.mean()`, at those stops. Today that is
available only through the ancestry record, whose memory grows with particles ×
compartments × steps, and as weighted summaries for `pfilter`
(`record_stream_means`). The change is an additive field on the prequential
record, filled where draws are. It draws no random number and changes no weight;
the test is that the likelihood, the draws and every weight are bit-identical
with it on and off. It touches `particle_filter.rs`, and is the only change in
this proposal that does.

**Identity.** A prediction's address folds: the parameter source (the fit's run
id, chain selection and draw count, or the resolved point); the model digest;
the mode with its origins and reach; the state set's row digests; each target's
design values; each covariate's statement and table digest (below); particles,
seed, step and sampler; and the engine build. Outcomes of predicted rows never
enter it.

### One table

Every mode writes the same file, `predictions.tsv`, one row per `PredictionRow`:

```
scenario stream <dims> t_start t_stop [date_start date_stop] mode origin
horizon lead role quantity parameters mean q05 q25 q50 q75 q95 n_draws
n_samples
```

`camdl predict` with no `--mode` writes free-forward and one-step rows into the
one table, as `fit predict` writes both today; the `mode` column tells them
apart. `forecast-samples.tsv` stays, with an `origin` column, as the joint
samples of D54. One manifest, `prediction.json`, replaces `predictive.json` and
`forecast.json`: the parameter source and set, the mode, the state conditioning,
the plans, the covariate statements, the calendar, the notes, and the streams
produced, skipped and failed (D76). One typed reader serves `camdl show` and
`camdl cat` (gh#1259, gh#1260).

A prediction's files hold only inputs to the prediction (D81). No observed value
and no likelihood of one appears in them, in any mode (D105).

### Covariates after the origin

Each forcing a forecast evaluates after its origin is in exactly one of five
named states, decided at plan time, before any filtering, and written in the
manifest per forcing and origin. "Evaluated after the origin" is judged on the
forcing's argument: for a forcing with a `lag`, `t − lag`.

```rust
pub enum AfterOrigin {
    /// Defined for all time by the model's own declaration: a formula, a
    /// periodic kind, or a piecewise function's last segment.
    ByDeclaration { kind: ForcingKind },
    /// Not evaluated after the origin within the reach (a lag at least as
    /// long as the reach, or a table read only before the origin).
    NotRead,
    /// Table values after the origin that the user states were fixed or
    /// assumed at the origin: a timetable, a planned campaign, a scenario.
    Prospective { table: TableRef },
    /// Table values after the origin that were observed after it.
    Realised { table: TableRef },
    /// The table cut to its rows at or before the origin, its last
    /// remaining value carried forward.
    Held { value: f64, from: f64 },
}
```

**Nothing is automatic for a table.** The first two states need nothing from the
user. The last three are never chosen by camdl: a ranged forcing that is read
after an origin and has no statement is refused, before any filter runs, with
the forcing, the date its table ends, the origin and the reach:

```
error: forcing `pop` is read after the forecast origin and nothing says how.
  origin 1961-01-01, reach 8 weeks (to 1961-02-26); its table `pop` has rows to 1964-12-31.
  State one of:
    --covariate pop=prospective   the values after the origin were fixed or assumed at the origin
    --covariate pop=realised      they were observed after the origin (an after-the-fact evaluation)
    --covariate pop=hold          carry the value at the origin forward
```

and when the table ends inside the reach, `prospective` and `realised` are not
offered and the message says the table ends on that date, short of the reach by
so many days, with `--table pop=FILE` and `hold` as the ways out.

The statement is a flag, `--covariate NAME=prospective|realised|hold`, and for a
fit it can be written once in the fit config, read by `predict` and `score` and
overridden by the flag:

```toml
[data.tables]
pop = "london_pop_1950_1964.tsv"

[forecast.covariates]
pop = "realised"
```

`[forecast.covariates]` is excluded from the fit's identity, as it cannot change
the fit, and is part of every prediction's (D108).

What each statement does:

- **`prospective`.** The table is read as it is. The forecast is a forecast,
  conditional on those values, and the manifest names the table and its digest.
- **`realised`.** The table is read as it is. The prediction and every score
  from it are labelled "conditional on covariates observed after the origin",
  and `compare` refuses to difference such a score against one that is not so
  labelled. This is the rule the forecast plan already applies to design columns
  read from held-out rows (`DesignMode::ExPost`, D81).
- **`hold`.** For each origin, the table is cut to its rows at or before the
  origin and the last remaining value is carried forward. It is the value a
  forecaster at the origin could have used, and with many origins each holds its
  own. It is refused for a cubic-spline table: a natural cubic spline is a
  global function of its knots, so cutting the table changes the forcing before
  the origin too, away from what the parameters were fitted under.

**A longer table for the prediction.** `predict --table NAME=FILE` supplies a
table longer than the one the fit stored, so a forecast can be made past the
fit's covariates without refitting. It is accepted only when the fitted forcing
is unchanged: every row of the fit's stored table is present and equal in the
new one, which may add rows only after the last. For a cubic-spline table that
is still a different function before the last row, so it is refused there, with
the advice to bind the full table at fit time. The new table's digest enters the
prediction's identity. A forcing bound in the model source (`data = "file.tsv"`)
is part of the model and cannot be extended this way; the refusal says to
declare it `external("NAME")`.

**Kinds that need no statement, and why that is safe.** A periodic forcing
(`periodic`, `sinusoidal`, `fourier`, `periodic_spline`) repeats after the
origin with no statement. Periodicity is the model's declaration, and the fit
already relied on it: a fit to ten years of data used the same school calendar
ten times. Continuing it adds no assumption that the parameters were not fitted
under. The same holds for a formula in `t`. Requiring a line in the fit config
would be a statement the user can only ever answer one way.

`piecewise` holds its last value for all time by its declared meaning (an open
last segment), which is the behaviour a table needs a `hold` statement for.
Because it is in the model's source and not a file of samples, it is treated as
`ByDeclaration`; because it is the one kind where a level is carried forward
with no word from the user, a forecast whose reach lies past the last breakpoint
prints one note naming the forcing, the breakpoint's date and the value carried.

An estimated `bspline` is defined only on its window. A forecast past the window
is refused as today; there is nothing to state, because no values exist and
carrying the end value of an estimated transmission rate forward is a modelling
decision that belongs in the model (gh#828).

The run-level `allow_forcing_extrapolation` is not consulted after a forecast
origin. It continues to govern `simulate`, `pfilter` and fits until gh#1300
replaces it; a fit made under it is still filtered under it up to the origin.

**The measles model as an example.** The He et al. fixture has three forcings:
`school`, periodic; `pop`, an interpolated table; and `birthrate`, an
interpolated table whose values are births four years earlier (the delay to
school entry). For a rolling forecast over 1961 to 1964 from parameters fitted
to 1950 to 1960: `school` needs nothing; `pop` needs a statement, and with the
population series recorded through 1964 the truthful one is `realised` (or
`hold`, to see what a forecaster then could have done); `birthrate`, if the
model applies the delay as a `lag` of four years, is `NotRead` for any reach up
to four years, because the births that matter were already recorded at the
origin. In the fixture the delay is applied in the data file instead
(`model.camdl`: "Birthrate lagged 4 years"), so camdl sees an ordinary table and
asks for a statement; `prospective` is the truthful one there, and the manifest
then rests on the user's word. Both tables are bound in the model source
(`data = "data/he2010_london_covariates.tsv"`), so they cannot be extended with
`--table` without becoming `external`. One further caution for that project: the
fixture's header says the monthly values were "pre-interpolated by R's
smooth.spline()". If that smoothing was fitted to the whole 1950 to 1964 series
(not checked), values before any origin already depend on data from after it,
which no statement here can see.

### Scores

#### What exists and what is missing

The `score` crate holds the per-row scores of a one-step pass
(`PrequentialTrace`) and its labels. `pfilter --save-prequential` writes one at
a fixed point. `compare` derives one per fit, by running `camdl pfilter` as a
child process for each of 64 posterior draws in each of 5 replicates, pooling,
differencing, and deleting the directory. For forecasts, the driver records each
scored target's per-particle likelihood and `forecast_score.rs` turns them into
log scores, which nothing calls.

Missing: a stored score; a score of a forecast; scores grouped by lead; a
one-step score that keeps training rows and marks them; a fixed-point score that
does not claim to be in-sample.

#### The command

```
camdl score FIT | MODEL --params FILE [--trained-through INSTANT]
    --mode one-step | forecast
    [--origin INSTANT ... | --origins SPEC] [--horizon DURATION]
    [--data SOURCE=FILE ...] [--covariate NAME=...] [--profile NAME]
```

`score` runs, or finds in the store, the prediction it names, with each scored
row's outcome bound as a held-out cell (D79): the filter evaluates the outcome's
likelihood per particle and never weights on it. It then writes the score
artifact, which records the prediction's address. The prediction is
bit-identical whether or not outcomes are bound (the existing D79 test), so
`predict` followed by `score`, and `score` alone, leave the same prediction
files.

For one-step mode the outcome of a row is also what the filter assimilates next,
so there the pass that predicts is the pass that scores. The prediction's files
still carry no outcome; the score's do.

#### The tables

`scored.tsv`, one row per scored row or target, is the plot-ready table:

```
scenario stream <dims> t_start t_stop [dates] mode origin horizon lead role
parameters mean q05 q25 q50 q75 q95 observed log_score crps pit_lo pit_hi
in_50 in_90
```

The quantiles are the observation quantity's. `role` here is `trained`,
`held_out` or `unknown`. `in_50` and `in_90` say whether the outcome fell inside
the central 50% and 90% intervals.

`summary.tsv`, one row per (stream, mode, role, lead):

```
stream <dims> mode role lead horizon n log_score_sum log_score_mean
crps_mean coverage_50 coverage_90
```

Scores at different leads, different roles or different modes are never summed
(D35). Both tables open with the fit's label, so tables from several models
stack by concatenation.

#### How each score is computed

The log score of a row is the log of the predictive density of its observed
value, estimated by the equal-weight mixture over the `N` particles of the `M`
draws: `logsumexp_{m,i} ℓ_{m,i} − log(MN)`, with `ℓ_{m,i}` the observation
model's log density at particle `i` of draw `m` (D79; `M = 1` for a fixed
point). It is never the mean of per-draw log scores, which by Jensen's
inequality is a smaller, different number.

- Under one-step mode, per draw, this is the conditional log likelihood of the
  row, the quantity pomp calls `cond_logLik`. Summed over rows at a fixed point
  it is the filter's log likelihood estimate, which is the test that ties
  `score` to `pfilter`.
- Under forecast mode the weights after the origin are equal by construction
  (nothing is assimilated and nothing resampled), so the same formula applies to
  each target, and the joint log score of an origin's targets sums `ℓ` over
  targets within a sample before the mixture.
- The draws are those of the posterior given the training rows and are not
  reweighted by held-out rows the state has since seen. That is what "no refit"
  means for a score (D35), and is the existing `Provenance::Posterior`
  definition.

CRPS, the PIT endpoints and the interval flags are computed from the pooled
observation draws by the `score` crate's existing kernels (the fair CRPS
estimator; the randomized PIT for counts).

#### The types, and how they fit together

A score carries the same descriptions as the prediction it scores, and nothing
that restates them in other words.

```rust
pub struct ScoreArtifact {
    pub schema: SchemaVersion,
    pub engine: EngineStamp,
    pub identity: ScoreIdentity,
    pub prediction: PredictionRef,        // the address of what was scored
    pub parameters: ParameterSource,      // as the prediction's
    pub conditioning: score::Conditioning,// the parameter set, per artifact
    pub mode: PredictionMode,             // the state set
    pub basis: PredictiveBasis,           // particle filter or ODE (exists)
    pub covariates: Vec<CovariateRecord>, // per forcing, its AfterOrigin
    pub time: TimeAxis,
    pub data: Vec<StreamData>,            // row digests of every scored row
    pub body: ScoreBody,
    pub summary: Vec<SummaryRow>,         // summary.tsv's rows
}

pub enum ScoreBody {
    OneStep(PrequentialTrace),
    Forecast(ForecastScores),
}

pub struct ForecastScores {
    /// Sorted by (origin, stream, end). Marginal scores only.
    pub targets: Vec<ScoredTarget>,
    /// Per origin, the joint log score of that origin's targets (D79).
    pub joint: Vec<JointScore>,
}
```

The score-artifact proposal defines a type `ScoredLeadTime` with variants
`OneObservationAhead` and `Unassimilated`, to say how far ahead of the last
assimilated observation each scored value was predicted. With the mode recorded,
it says nothing more: `OneObservationAhead` is `OneStep` on a filtered basis and
`Unassimilated` is `FreeForward`. It is not built, and this proposal drops it in
favour of `mode` and a `lead` on every scored row; the per-stream `LeadSpan`
(the smallest and largest lead among a stream's scored rows) is kept, because
"one step" is a week for a weekly stream and a day for a daily one.

Held-out outcomes enter the score's identity (its `data` level folds the row
digest of every scored row) and never the prediction's (D81).

`compare` reads score artifacts and refuses to difference two whose mode,
origins, leads, scored rows, conditioning, basis or covariate statements differ,
each refusal naming the field. The checks it has today (`score::preflight`) are
the same refusals over the trace's fields.

#### What changes in the score-artifact proposal

`2026-11-14-score-artifact.md` (proposal A) was written before the temporal
migration and before the forecast driver.

| Proposal A                                                                          | State on `main`                                                                                   | Under this proposal                                                                                |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| A2: `observed_rows`, `row_digest`                                                   | built by the migration (`fit/observed_rows.rs`, `fit/training_partition.rs`)                      | done; A2 is struck                                                                                 |
| A3: `train_end`, `training_rows`, the held-out rule                                 | built as `TrainingPartition` and the spec's §4.2 rule                                             | A3 keeps only the three refusals and the archived-model requirement                                |
| A4: `HASH_VERSION` 4; row digests in the fit identity; holdout files out of it      | `HASH_VERSION` is 7; the fit identity still holds whole-file and holdout-file digests (spec §5.4) | the identity step is still owed, as its own bump (D58)                                             |
| A1: trace schema 6                                                                  | `main` already writes schema 6 (_run_)                                                            | the trace change is schema 7                                                                       |
| A1: `PassTrace` and `pfilter --save-pass`, one child process per draw per replicate | not built; the one-step band and the forecast driver already run the filter in process            | scores are derived in process; `PassTrace` stays in memory and `--save-pass` is not added (D104)   |
| `lead: ScoredLeadTime`                                                              | not built                                                                                         | dropped; `mode: PredictionMode` and per-row `lead`                                                 |
| `Conditioning`: `HoldOutTail`, `HoldOutTailUnproven`, `InSample`                    | `InSample`, `HoldOutTail`                                                                         | adds `HoldOutDeclared` and `Unknown`; `HoldOutTailUnproven` as A specifies                         |
| `trace: PrequentialTrace` as the only body                                          | not built                                                                                         | `body: ScoreBody`                                                                                  |
| `fit: ScoredFit`                                                                    | not built                                                                                         | `parameters: ParameterSource`, so a fixed point is a scorable thing                                |
| settings level of the identity                                                      | not built                                                                                         | also folds the mode, the origins, the reach, the covariate statements and the prediction's address |

## Before and after

A fit on 1950 to 1960, with 1961 to 1964 held out by `holdout_after`:

```bash
# before: three tables in two layouts, no scores kept
camdl fit predict @sir                     # free-forward + one-step + a tail
camdl fit predict @sir --horizon "4 weeks" # one origin, the training extent
camdl compare @sir @seir                   # one-step scores, not stored

# after
camdl predict @sir                         # free-forward + one-step, one table
camdl predict @sir --mode forecast --horizon "8 weeks" \
    --origins "every 4 weeks from 1961-01-01 to 1964-10-01" --covariate pop=realised
camdl score   @sir --mode forecast --horizon "8 weeks" \
    --origins "every 4 weeks from 1961-01-01 to 1964-10-01" --covariate pop=realised
```

A published parameter vector, no fit:

```bash
# before: one-step only, through pfilter; rows before the cut are dropped
camdl pfilter measles.camdl --params he2010.toml --data cases=london.tsv \
    --particles 2000 --save-prequential he --score-from 1961-01-01

# after: every row, each marked, with its log score
camdl score measles.camdl --params he2010.toml --data cases=london.tsv \
    --trained-through 1961-01-01 --mode one-step --particles 2000
```

`scored.tsv` from the second, abridged (values illustrative):

```
stream t_start    t_stop     mode     origin     lead role     parameters  q05 q50 q95 observed log_score
cases  1960-12-18 1960-12-25 one_step 1960-12-18 7    trained  fixed_point 212 301 420 288      -5.91
cases  1960-12-25 1961-01-01 one_step 1960-12-25 7    trained  fixed_point 230 322 451 340      -6.02
cases  1961-01-01 1961-01-08 one_step 1961-01-01 7    held_out fixed_point 247 349 488 402      -6.37
```

Its manifest says: "parameters: a fixed point from he2010.toml; no parameter
uncertainty. Training rows: declared by --trained-through, not verified."

Removed spellings fail with a message naming the replacement and are not kept as
aliases.

## What this proposal does not do

- It does not refit at each origin. Parameters are fixed from one fit (D35).
- It does not filter once and copy particles at each origin (D102).
- It does not make `simulate --init-state` exact inside a period (PR #1158).
- It does not add a per-forcing hold to the model language (gh#1300), or let an
  estimated spline continue past its window (gh#828).
- It does not import or export Hubverse files, or score forecasts made outside
  camdl (D37).
- It does not add aggregated targets (weekly targets from a daily stream), the
  calendar `covers` forms (D45), or data vintages.

## Stages

Each stage is one pull request with its tests and a `Proposal:` footer, and each
leaves `main` consistent.

1. **Finish Stage 5.** Delete the free-forward tail (`forecast_times`,
   `leaf_row_coverages`; gh#1151, gh#1152). Rename the `horizon` column that
   holds a mode to `mode` (gh#1302). Update the spec's "Not yet built" markers
   for what Stage 5 built (gh#1303).
2. **The row and the table.** `PredictionRow`, `Quantity`, `RowRole`,
   `StateConditioning::{None, Sequential}`; `predictions.tsv` and
   `prediction.json` written by all three modes; `convergence.tsv`; the shared
   reader (gh#1259, gh#1260); one row reader, so one-step predicts
   `[data.holdout]` rows (gh#1309). The per-particle mean on the prequential
   record, and `--quantity`.
3. **Parameter sources.** `ParameterSource`, the `ParameterSet` enum;
   `--params`, `--trained-through`; single-estimate fits accepted and labelled
   (D101); `score::Conditioning::{HoldOutDeclared, Unknown}` (gh#1305). The
   command becomes `camdl predict` (D100).
4. **Covariates after the origin.** `AfterOrigin`, the plan-time classification
   and refusal with dates (gh#1304), `--covariate`, `[forecast.covariates]`,
   `predict --table`, the manifest record (gh#1083) (D108).
5. **Origins.** The origin and state set as arguments of `ForecastPlan::new`;
   `--origin`, `--origins`; one plan and one driver call per origin; the
   measured cost (D102).
6. **Forecast scores (the runbook's Stage 6).** `ScoreArtifact` with
   `ScoreBody::Forecast`, `camdl score --mode forecast`, `scored.tsv`,
   `summary.tsv`. Proposal A's steps A6 and A7, amended as above.
7. **One-step scores through the same command.** `camdl score --mode one-step`,
   in process (D104); `compare` reads stored scores (A8). The fit-identity step
   (A4) lands here, as its own `HASH_VERSION` bump, after confirmation.
8. **Trim `pfilter`.** Remove its predictive outputs once stage 7 gives the same
   numbers (D103).

Stage 1 needs no decision beyond approval of this proposal. Stage 2 needs D105,
stage 3 D100 and D101, stage 4 D108, stage 5 D102, stage 7 D104, stage 8 D103.
Covariates come before origins because many origins multiply every unstated
covariate read.

## Tests

Each is written to fail before its stage.

- **One table, three modes.** A fit with a mid-series `holdout_after`: every row
  of `predictions.tsv` has all key columns; one-step rows after the cutoff are
  `held_out`; forecast targets are `future`; no free-forward row lies past the
  data.
- **The two ways of holding out agree.** The same split declared by
  `holdout_after` and by a `[data.holdout]` file gives identical one-step rows.
- **Quantity.** For a Poisson stream at a fixed point, the expectation band's
  variance plus the mean of the expectation equals the observation band's
  variance within Monte Carlo error; for a model with no observation noise the
  two bands coincide.
- **Recording the expectation changes nothing.** The filter's likelihood, every
  weight and every draw are bit-identical with the per-particle mean recorded
  and not.
- **Fixed point equals a one-draw posterior.** `predict --params P` and a
  posterior whose every draw is `P` give identical `predictions.tsv` apart from
  the `parameters` column.
- **Fixed point is labelled.** The manifest and the table say `fixed_point`;
  with no `--trained-through`, every role is `unknown` and the score's
  conditioning is `unknown`; with it, `hold_out_declared`.
- **One-step score equals the filter's likelihood.** At a fixed point and
  matched seed, the sum of `scored.tsv`'s one-step `log_score` over all rows
  equals `pfilter`'s log likelihood exactly.
- **An origin later than the training extent.** The spec §4.4 reference model
  (daily and weekly incidence): with the origin two weeks past the training
  extent, the predictive mean and variance of the weekly flow agree with an
  importance-sampling reference that conditions on the daily rows through the
  new origin. With the origin at the training extent the forecast is today's,
  bit for bit.
- **No leak per origin.** For each origin, changing any row ending after it
  leaves that origin's predictions and address unchanged; changing a row ending
  before it changes both.
- **Rolling equals repeated single origins.** A run with three origins equals
  three single-origin runs, row for row.
- **Outcomes never reach a prediction.** Changing held-out outcomes changes
  `scored.tsv` and leaves `predictions.tsv`, `forecast-samples.tsv` and the
  prediction's address unchanged (D79, D81).
- **Scores are never pooled.** `summary.tsv` has one row per (stream, mode,
  role, lead); `compare` refuses two artifacts whose origins or leads differ.
- **Coverage.** On data simulated from the model at the scoring parameters,
  `coverage_50` and `coverage_90` over many origins fall within binomial error
  of 0.5 and 0.9 at each lead.
- **A covariate with no statement is refused before any filtering,** naming the
  forcing, the date its table ends, the origin and the reach; the same model
  with a periodic forcing only is not asked anything.
- **Each statement does what it says.** `hold` at an origin equals a run whose
  table file was cut at that origin by hand; `realised` equals today's forecast
  and is labelled; a lagged forcing whose lag exceeds the reach is `not_read`.
- **A realised-covariate score is not compared with a prospective one.**
- **A longer table that changes the fitted forcing is refused,** including any
  cubic-spline table.
- **The matrix.** For each fit method and backend the runbook's Stage 5 table
  lists, each mode is either produced or refused with its message; the two cells
  this proposal changes (IF2 and NLopt fits) are produced and labelled.
- **Removed spellings.** Each fails naming its replacement.

## References

- Bosse NI, Gruson H, Cori A, van Leeuwen E, Funk S, Abbott S. Evaluating
  forecasts with scoringutils in R. arXiv:2205.07090. The package manual,
  version 2.3.0, pages `as_forecast_quantile`, `score`, `summarise_scores`,
  `get_metrics.forecast_quantile`, `logs_sample`:
  <https://epiforecasts.io/scoringutils/reference/>.
- Dawid AP (1984). Statistical theory: the prequential approach. _Journal of the
  Royal Statistical Society A_ 147(2):278–292. doi:10.2307/2981683. Cited
  through Gneiting and Raftery; not read for this proposal.
- Gneiting T, Raftery AE (2007). Strictly proper scoring rules, prediction, and
  estimation. _Journal of the American Statistical Association_
  102(477):359–378. doi:10.1198/016214506000001437. §7.1.
- Hubverse documentation, user guide, "Model output", "Tasks", "Target
  (observed) data" and "Sample output type": <https://docs.hubverse.io/>.
- Hyndman RJ, Athanasopoulos G (2021). _Forecasting: Principles and Practice_,
  3rd ed. OTexts. §5.8, §5.10. <https://otexts.com/fpp3/>.
- Jordan A, Krüger F, Lerch S (2019). Evaluating probabilistic forecasts with
  scoringRules. _Journal of Statistical Software_ 90(12).
  doi:10.18637/jss.v090.i12. §2.2.
- King AA, Nguyen D, Ionides EL (2016). Statistical inference for partially
  observed Markov processes via the R package pomp. _Journal of Statistical
  Software_ 69(12). doi:10.18637/jss.v069.i12. Not re-read for this proposal;
  the claims about pomp rest on the manual, version 6.5, pages `simulate`,
  `pfilter`, `forecast`, `covariate_table`:
  <https://kingaa.github.io/pomp/manual/>, and on the SBIED course's Ebola
  lesson: <https://kingaa.github.io/sbied/ebola/>.
