# Proposal: one predictive command, one prediction table, one score command

**Status:** Draft for the maintainer's review. Not approved; nothing here is to
be implemented until it is. **Date:** 2026-11-19. **Depends on:**
`docs/dev/temporal-architecture.md` (the temporal spec) and
`2026-11-15-temporal-architecture-migration.md` (its runbook), whose Stage 5 is
on `main`. **Amends:** `2026-11-14-score-artifact.md` (the sections listed under
[The score artifact](#the-score-artifact)) and the runbook's Stage 6.
**Supersedes:** `2026-08-29-honest-predictive-evaluation.md` §3.7.2, §3.8 and
its Stage 5. **Decisions it assumes:** D100 to D105 in the maintainer's docket,
each marked where it is used; the draft takes the recommended option of each.

Code references are to `main` at `c92e7c9e3`. A statement marked _run_ was
checked by running the release binary built from that commit; one marked _read_
was read in the code and not run.

## Summary

camdl answers "what does this model predict?" in four places, with four table
layouts and three different rules for what the prediction was allowed to see.
This proposal gives prediction one command, `camdl predict`, with one typed
description of a prediction (where the parameters came from, what the state was
conditioned on, which quantity each band shows) and one table layout. Scoring is
a second command, `camdl score`, which is the only thing that reads the outcomes
a forecast did not see. `pfilter` keeps the likelihood at one parameter vector
and its diagnostics, and `simulate` stays a generator with no data.

It adds four capabilities a downstream model comparison needs: predicting from a
fixed parameter vector with no fit; one-step-ahead prediction through held-out
rows with every row marked; forecasts from many forecast origins scored by lead
time; and a single plot-ready table over training and held-out rows.

## Terms

Terms from the temporal spec are used as defined there (§2.2 to §2.6, §5.5) and
restated in one line each.

- **Stream.** One concrete observation series of a model's `observations {}`
  block.
- **Period, end.** What a data row measures (a count over a stretch of time) and
  the instant that stretch ends, which is when the row is scored.
- **Training rows, training extent.** The rows a fit's parameters were fitted
  to, and the latest end among them. Rows that did not train are **held out**.
- **Parameter set.** The rows the parameters were fitted to.
- **State set.** The rows the latent state was conditioned on when a prediction
  was made.
- **Forecast origin.** The instant after which a forecast sees no data.
- **Target.** One stream's period that a forecast predicts.
- **Lead time.** The duration from the forecast origin to the target's end.
- **Horizon.** A target's position among its stream's targets, counting the
  first one that ends after the origin as 0. On the command line,
  `--horizon "4 weeks"` is instead a duration: how far past the origin the
  forecast reaches.
- **Fixed point.** One parameter vector supplied by the user or taken from a
  single-estimate fit, as opposed to draws from a posterior.
- **Expectation, observation.** Two quantities a band can show. The
  _expectation_ of a row is `E[y | x, θ]`, the observation model's mean given
  the latent state `x` and parameters `θ`; its spread is uncertainty about the
  state and the parameters only. The _observation_ is a draw of `y` itself, with
  the observation model's noise added.

## What exists today

### Four commands predict

| Command                                       | Parameters                       | What the state has seen                                                         | Output                                                                            |
| --------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `simulate`                                    | one vector, or draws (`--draws`) | nothing                                                                         | trajectories; `--obs` files                                                       |
| `simulate --init-state FILE`                  | one vector                       | whatever the filter that wrote `FILE` saw                                       | the same, from the saved time on                                                  |
| `simulate --init-state fit --draws posterior` | a PGAS fit's draws               | each draw's own sampled path                                                    | the same                                                                          |
| `pfilter --save-prequential`                  | one vector                       | each row: every earlier row                                                     | `STEM.streams.tsv`: observed value, band of the observation, log score, CRPS, PIT |
| `pfilter --save-filtered-streams`             | one vector                       | each row: every earlier row (`predictive`), and the row itself too (`filtered`) | band of the expectation                                                           |
| `fit predict --mode free_forward`             | posterior draws                  | nothing                                                                         | `predictive/<stream>.tsv`: band of the observation, plus a tail past the data     |
| `fit predict --mode one_step`                 | posterior draws                  | each row: every earlier row                                                     | the same file, more rows                                                          |
| `fit predict --mode forecast --horizon D`     | posterior draws                  | the training rows                                                               | `forecast/<stream>.tsv`, `forecast-samples.tsv`, `forecast.json`                  |
| `compare`                                     | a fit's draws                    | each scored row: every earlier row                                              | one-step scores, computed in a temporary directory and not kept                   |

What this costs a user, each item checked as marked:

- **A fixed parameter vector cannot be forecast.** `fit predict --params p.toml`
  is refused by the argument parser (_run_: "unexpected argument '--params'"),
  and a single-estimate fit (IF2, the NLopt optimizers) is refused by decision
  D51 (`fit/predict.rs:4501`, _read_). The one-step predictive at a fixed vector
  exists only under `pfilter`, and a forecast from one only through
  `simulate --init-state`, which carries no targets, lead times or record of
  what the state saw.
- **The same quantity has two names and two layouts.** `pfilter`'s
  `STEM.streams.tsv` has columns
  `t stream y_obs y_pred_q05 … y_pred_q95 log_score crps pit`; its
  filtered-streams file has `time stream phase mean q05 … q95`, where `phase` is
  `predictive` or `filtered` (_run_). `fit predict` writes
  `scenario time <dims> horizon treatment … q05 … q95`, where the `horizon`
  column holds the mode (`free_forward`, `one_step`), while the forecast table's
  `horizon` column holds an integer (`fit/predict.rs:2919`,
  `fit/forecast_output.rs:632`, _read_).
- **No table states which quantity its band shows.** `fit predict`'s bands are
  bands of the observation; `--save-filtered-streams` bands are bands of the
  expectation; neither file says so in a column, and the expectation band is not
  available for a posterior at all.
- **Held-out rows were marked nowhere in a prediction.** `fit predict` one-step
  filters through rows after `holdout_after`; since the fix for gh#1079
  (`7918f460a..ec75f4c66`) each row of `predictive/<stream>.tsv` carries a
  `role` (`trained`, `held_out`, `future`) and the manifest names both
  information sets, which is the marking this proposal carries into every mode.
  A fit with separate holdout files still gets no one-step row for them
  (gh#1309). `pfilter --score-from 21` drops the rows before day 21 from the
  trace rather than marking them, so one table over training and held-out rows
  cannot be had (_run_).
- **There are two forecasts in one command.** `fit predict` with no flags writes
  a free-forward tail past the last row, on a grid built from the most common
  gap between labels, starting at the largest label of the whole file including
  held-out rows (`fit/predict.rs:2161`, `forecast_times`, _read_; gh#1151,
  gh#1152, both open). The runbook's Stage 5 lists this code for deletion; it is
  still there, beside the plan-based forecast mode.
- **Forecast scores are computed and thrown away.** The fixed-origin driver
  binds each target's held-out value and evaluates the per-particle likelihood
  (D79); the log-score functions exist and have no caller outside tests
  (`fit/forecast_score.rs:31`, _read_). `camdl score` does not exist (the
  command enum has no such variant, `main.rs:166-293`, _read_).
- **Only one forecast origin exists.** It is the training extent (D57). Scoring
  by lead time needs many.

### What a restart is, and is not

`pfilter --save-final-state` writes the particles' compartment counts at the
last row; `simulate --init-state` continues from them. It restores counts and
nothing else, so it cannot produce the row of a reporting period under way at
the restart. Since the fix for gh#1081 (`8e404048c..0a377eb28`) it refuses to
write observations in that case, naming the stream and the period. Making it
exact is PR #1158's subject and is not needed by anything below: no prediction
in this proposal goes through a saved state.

## How others divide the same work

The three systems below were read in their current documentation; the report
with page-level sources is in the references.

- **pomp** (King, Nguyen and Ionides 2016; manual 6.5) keeps three functions.
  `simulate` draws from the model with no conditioning on data. `pfilter` runs
  at one parameter vector and returns, per observation time, the conditional log
  likelihood (`cond_logLik`), the effective sample size, and optionally the
  prediction and filtering means of the _state_ (`pred.mean`, `filter.mean`).
  `forecast()` returns the observation mean at the mean predicted state, a
  plug-in number, not a predictive distribution. Forecasting past the data is
  user code: the Ebola lesson of the SBIED course saves the final particles,
  writes them into initial-value parameters, and calls `simulate`. So pomp has
  no object for a forecast, no forecast origin or lead time, and no marking of
  rows as training or held-out. camdl's `simulate --init-state` is that idiom,
  with its limits.
- **The Hubverse** (hubverse.io documentation, schema 6.0) defines a forecast as
  rows of one table: task-identifying columns (`origin_date`, `horizon`,
  `target`, `location`, and the derived `target_end_date`), then `output_type`
  (`mean`, `median`, `quantile`, `cdf`, `pmf`, `sample`), `output_type_id` and
  `value`. Observations live elsewhere, in `target-data/`, and are joined to
  model output only when scoring; the documentation notes that the data version
  a forecast was made from and the one it is scored against are different
  things. Samples sharing an `output_type_id` are one joint draw across
  horizons.
- **scoringutils** (Bosse et al.; manual 2.3) scores a table that holds
  `observed` and `predicted` side by side, keyed by a _forecast unit_ (the
  columns that identify one forecast), and summarises with
  `summarise_scores(by = …)`, for example by model and horizon. Its defaults for
  quantile forecasts include the 50% and 90% interval coverage. Its sample-based
  log score is a kernel density estimate that its manual says is for continuous
  predictions only; Jordan, Krüger and Lerch (2019, §2.2) recommend a mixture
  over parameter draws of the analytic density instead, which is what a particle
  filter's conditional likelihood is.

The design takes one thing from each. From pomp, the split between a generator
with no data, a likelihood at one vector, and prediction as something built on
them. From the Hubverse, the prediction table keyed by origin, target and
horizon, with the quantity named in a column and the observations kept out of
it. From scoringutils, the score table that holds observed and predicted
together and is summarised by any grouping, with coverage at 50% and 90%. The
log score follows Gneiting and Raftery (2007, §7.1): for time-ordered data the
sum of one-step log predictive densities is the prequential log likelihood of
Dawid (1984), so it is computed from the filter's own densities and never from a
density estimate over draws.

The names for forecast evaluation follow Hyndman and Athanasopoulos
(_Forecasting: Principles and Practice_, 3rd ed., §5.8 and §5.10): errors on
training rows are residuals and not forecast errors, and evaluation on a rolling
forecasting origin is what scoring from many origins is called there.

## The design

### Division of labour

| Command    | What it is for                                                                  | Reads observed data                         |
| ---------- | ------------------------------------------------------------------------------- | ------------------------------------------- |
| `simulate` | generate from the model                                                         | never                                       |
| `pfilter`  | the likelihood at one parameter vector, and diagnostics of that filter run      | every bound row                             |
| `predict`  | predictions: bands and samples, each with what conditioned it                   | only the rows in the prediction's state set |
| `score`    | scores of a prediction against outcomes; the plot-ready table with observations | the outcomes of every scored target         |
| `compare`  | differences between stored scores of several fits                               | nothing: it reads score artifacts           |

`simulate --init-state` remains as a lower-level way to continue a simulation
from saved counts (D64). Its documentation stops calling it forecasting.

`camdl predict` replaces `camdl fit predict` (D100). A prediction from a fixed
point has no fit, so the command cannot sit under `fit`.

### Types

Three questions describe a prediction, and each becomes one type. They extend
the forecast plan of the spec's §3.3; `ForecastPlan`, `Target` and
`TrainingPartition` are the existing types (`fit/forecast_plan.rs`,
`fit/training_partition.rs`).

**Where the parameters came from.**

```rust
pub enum ParameterSource {
    /// Draws of a sampler fit (PMMH, PGAS, MH on ODE): a mixture over draws.
    Posterior { fit: FitRef, chains: ChainSelection, n_draws: usize },
    /// One parameter vector. The prediction carries no parameter uncertainty.
    FixedPoint { point: ParamPoint, from: PointOrigin },
}
pub enum PointOrigin {
    /// `--params FILE`. camdl knows nothing of how it was obtained.
    File { digest: ContentHash },
    /// The single estimate of an IF2 or NLopt fit.
    FitEstimate { fit: FitRef },
}
```

This replaces `ParamTreatment` (`fit/predict.rs:72`), whose `PlugIn` variant
exists only to be refused. The table column `parameters` holds `posterior` or
`fixed_point`; a fixed-point manifest carries the sentence "fixed point: no
parameter uncertainty", and every message that prints a band from one repeats
it. Allowing it reverses D51 (D101).

**Which rows the parameters were fitted to, and how camdl knows.**

```rust
pub enum ParameterSet {
    /// Read from a fit's record and verified against the data by row digest.
    Recorded(TrainingPartition),
    /// Stated by the user for a fixed point (`--trained-through TIME`).
    /// camdl cannot check it; the manifest says "declared, not verified".
    Declared(TrainingPartition),
    /// A fixed point with nothing stated.
    Unknown,
}
```

A fixed vector taken from a paper was fitted to some data, and camdl cannot know
which. Under `Unknown` every row is marked `unknown` and no score is stamped
held-out. Under `Declared` rows are marked from the stated cutoff with the same
rule a fit uses (a row trains if its period ends at or before the cutoff, spec
§4.1), and every artifact records that the partition was declared. A score from
a declared partition is labelled "held-out as declared", a third value beside
the existing in-sample and held-out, so it can never be read as verified.

**What the state had seen when each row was predicted.**

```rust
pub enum Conditioning {
    /// Nothing: the model run forward from its start.
    FreeForward,
    /// Each row is predicted from every row, of every stream, that ends
    /// before it; the filter then assimilates it.
    OneStep,
    /// At each origin the state has seen the rows ending at or before it,
    /// and nothing after.
    Forecast { origins: Origins, reach: Duration },
}
pub enum Origins {
    /// The training extent. Today's `--mode forecast`.
    TrainingExtent,
    /// Listed instants (`--origin`, repeatable).
    At(Vec<Instant>),
    /// A regular sequence (`--origins "every 4 weeks from T0 to T1"`).
    Every { step: Duration, from: Instant, to: Instant },
}
```

`Conditioning` replaces the `Horizon` enum (`fit/predict.rs:38`) and the
separate entry into forecast mode. A forecast from several origins is a list of
the existing `ForecastPlan`, one per origin, each built by the existing
`ForecastPlan::new`; the plan gains the field the spec sketched and the code
omitted:

```rust
pub enum StateConditioning {
    None,
    /// Per stream, the rows assimilated: a row prefix with its digest.
    Filtered { through: f64, rows: Vec<RowPrefix> },
}
```

The parameter set and the state set differ whenever an origin is later than the
training extent: the parameters stay those of the one fit, and the state is
filtered through the newer rows without a refit (D35; this is the work D57
deferred, taken up as D102). Three checks in the constructor:

- an origin earlier than the first row of every stream is refused;
- at each origin, no row ending after it is bound as an observation (the
  existing driver invariant, per origin);
- the state set's row prefixes agree with the parameter set on the rows they
  share.

An origin inside the training window is allowed. Its targets are rows the
parameters were fitted to, so they are marked `trained` and are residual-like in
the sense of Hyndman and Athanasopoulos: useful for seeing how the model tracks
the series it was fitted to at four weeks' lead, and never pooled with held-out
targets.

**Which quantity a band shows.**

```rust
pub enum Quantity {
    /// E[y | x, θ] per particle and draw: no observation noise.
    Expectation,
    /// A draw of y from the observation model.
    Observation,
}
```

Both are already computed: the free-forward replay keeps each draw's expected
value beside its `y_rep` draw (`fit/predict.rs:1179`), and the filter records
each stream's conditional mean for `--save-filtered-streams`.
`--quantity
observation|expectation|both` selects; the default is `both`, so a
table always shows the two side by side and a plotted band cannot be mistaken
for the other.

**The row.**

```rust
pub enum RowRole { Trained, HeldOut, Future, Unknown }

pub struct PredictionRow {
    pub scenario: String,
    pub stream: String,
    pub stratum: Vec<(String, String)>,
    pub measures: Measures,          // the period or instant predicted
    pub mode: Mode,                  // free_forward | one_step | forecast
    pub origin: f64,                 // see below
    pub horizon: Option<i64>,        // forecast only
    pub lead: f64,                   // measures.end() − origin
    pub role: RowRole,
    pub quantity: Quantity,
    pub parameters: ParameterKind,   // posterior | fixed_point
    pub mean: f64,
    pub quantiles: Quantiles,        // the existing fixed set, q05 … q95
    pub n: usize,                    // draws × particles pooled
}
```

`origin` is defined in every mode, so `lead` is too: the run's start under
free-forward; under one-step, the end of the latest row of any stream
assimilated before this row, which varies by row (spec §2.6); the forecast
origin under forecast.

The column is named `role`, the spec's word (§5.3) and the name the predictive
table already has on `main`. `phase` is not used for it: `pfilter`'s
filtered-streams file uses that word for before or after assimilation.

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

### One table

Every mode writes the same file, `predictions.tsv`, one row per `PredictionRow`:

```
scenario stream <dims> t_start t_stop [date_start date_stop] mode origin
horizon lead role quantity parameters mean q05 q25 q50 q75 q95 n
```

It replaces `predictive/<stream>.tsv` and `forecast/<stream>.tsv`.
`forecast-samples.tsv` stays, with an `origin` column, as the joint samples of
D54. One manifest, `prediction.json`, replaces `predictive.json` and
`forecast.json`: the three types above, the plans, the calendar, the notes, and
the streams produced, skipped and failed (D76). It is read by one typed reader
shared by `camdl show` and `camdl cat` (gh#1259, gh#1260).

A prediction's files hold only inputs to the prediction (D81). No observed value
and no likelihood of one appears in them, in any mode (D105); the table with
observations is the score's. The convergence columns of today's file
(`fit_rhat_max`, `rhat_mean`, …) move to a sibling `convergence.tsv` keyed the
same way, since they describe the fit's chains and not the prediction, and are
empty for a fixed point.

### One score command

```
camdl score FIT|--params FILE  --mode one-step | forecast
    [--origin T ... | --origins SPEC] [--horizon DURATION]
    [--data SOURCE=FILE ...] [--profile NAME] [--in-sample]
```

`score` runs, or finds in the store, the prediction it names, with each scored
target's outcome bound as a held-out cell (D79): the filter evaluates the
outcome's likelihood per particle and never weights on it. It then writes the
score artifact, which records the prediction's address. The prediction is
bit-identical whether or not outcomes are bound (the D79 test), so `predict`
followed by `score` and `score` alone give the same prediction files.

The score artifact writes two tables beside `score.json`.

`scored.tsv`, one row per scored target, which is the plot-ready table:

```
scenario stream <dims> t_start t_stop [dates] mode origin horizon lead role
parameters mean q05 q25 q50 q75 q95 observed log_score crps pit_lo pit_hi
in_50 in_90
```

The quantiles are the observation quantity's (an outcome is compared with the
predictive distribution of `y`, not of its mean). `role` here is `trained` or
`held_out`. `in_50` and `in_90` say whether the outcome fell inside the central
50% and 90% intervals.

`summary.tsv`, one row per (stream, mode, role, lead):

```
stream <dims> mode role lead horizon n log_score_sum log_score_mean
crps_mean coverage_50 coverage_90
```

Rows are grouped by lead time, and a stream with a declared cadence carries the
matching horizon. Scores at different leads, different roles or different modes
are never summed (D35). Both tables open with the fit's label, so tables from
several models stack by concatenation; `compare` reads the artifacts and refuses
to difference scores whose mode, origins, leads or scored rows differ.

The log score of a target is `logsumexp_{m,i} ℓ_{m,i} − log(MN)` over the `N`
particles of the `M` draws (D79), with `M = 1` for a fixed point. Under one-step
mode this is the conditional log likelihood of the row, the quantity pomp calls
`cond_logLik`; summed over rows at a fixed point it is the filter's log
likelihood estimate, which is the test that ties `score` to `pfilter`.

### The score artifact

`2026-11-14-score-artifact.md` (proposal A) was written before the temporal
migration and before the forecast driver. This is what changes in it.

| Proposal A                                                                          | State on `main`                                                                                             | Under this proposal                                                                                                    |
| ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| A2: `observed_rows`, `row_digest`                                                   | built by the migration (`fit/observed_rows.rs`, `fit/training_partition.rs:59`)                             | done; A2 is struck                                                                                                     |
| A3: `train_end`, `training_rows`, the held-out rule                                 | built as `TrainingPartition` and the spec's §4.2 rule                                                       | A3 keeps only the three refusals and the archived-model requirement                                                    |
| A4: `HASH_VERSION` 4; row digests in the fit identity; holdout files out of it      | `HASH_VERSION` is 7; the fit identity still holds whole-file and holdout-file digests (spec §5.4)           | the identity step is still owed, as its own bump (D58)                                                                 |
| A1: trace schema 6                                                                  | `main` already writes schema 6 (_run_)                                                                      | the trace change is schema 7                                                                                           |
| A1: `PassTrace` and `pfilter --save-pass`, one child process per draw per replicate | not built; the one-step band and the forecast driver already run the filter in process (`bootstrap_filter`) | scores are derived in process by the driver; `PassTrace` stays an in-memory type and `--save-pass` is not added (D104) |
| `lead: ScoredLeadTime` with `OneObservationAhead`, `Unassimilated`                  | not built                                                                                                   | gains `FixedOrigin { origins }`; an artifact holds one mode                                                            |
| `trace: PrequentialTrace` as the only body                                          | not built                                                                                                   | `body: ScoreBody`, either the trace (one-step) or `ForecastScores` (per origin and target)                             |
| `fit: ScoredFit`                                                                    | not built                                                                                                   | `parameters: ParameterSource` with `parameter_set: ParameterSet`, so a fixed point is a scorable thing                 |
| settings level of the identity                                                      | not built                                                                                                   | also folds the mode, the origins, the reach and the prediction's address                                               |
| "the word horizon is not used"                                                      | D41 fixed its two meanings                                                                                  | uses horizon and lead as the spec defines them                                                                         |

```rust
pub enum ScoredLeadTime {
    OneObservationAhead,
    Unassimilated,
    /// Each target is predicted from a state that saw rows up to its origin.
    FixedOrigin { origins: Vec<Instant> },
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

Held-out outcomes enter the score's identity (its `data` level folds the row
digest of every scored row) and never the prediction's (D81).

### Covariates over the reach

A forecast that runs past the last value of a data-table forcing is refused
today, naming the forcing, its range and the side (spec §4.6). Three gaps
remain, and each has its own issue; none is designed here.

- The refusal prints the range in model time (`t = 3650`) even for a model with
  a calendar origin, where the date the forcing runs out is what a user needs.
- A forecast that reads realised covariate values from after its origin (births
  recorded in 1962 used by a forecast made from 1960) is not labelled (gh#1083).
  This proposal's `prediction.json` has the `notes` field that label goes in.
- There is no way to give a prediction a longer covariate table than the fit
  had. The fit binds tables by name (`[data.tables]`, `external("NAME")`), so
  `predict --table NAME=FILE` is the natural form; it must be refused unless the
  new table agrees with the fit's over the span before the earliest origin,
  since the parameters were fitted under those values. Whether a covariate may
  be held past its range is the per-forcing declaration of gh#1300.

## Before and after

A fit on 1950 to 1960, with 1961 to 1964 held out by `holdout_after`:

```bash
# before: three tables in two layouts, held-out rows unmarked, no scores kept
camdl fit predict @sir                     # free-forward + one-step + a tail
camdl fit predict @sir --horizon "4 weeks" # one origin, the training extent
camdl compare @sir @seir                   # one-step scores, not stored

# after
camdl predict @sir --mode one-step
camdl predict @sir --mode forecast --origins "every 4 weeks from 1961-01-01 to 1964-12-01" \
    --horizon "8 weeks"
camdl score   @sir --mode forecast --origins "every 4 weeks from 1961-01-01 to 1964-12-01" \
    --horizon "8 weeks"
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

`scored.tsv` from the second, abridged:

```
stream t_start    t_stop     mode     origin     lead role    parameters  q05 q50 q95 observed log_score
cases  1960-12-18 1960-12-25 one_step 1960-12-18 7    trained  fixed_point 212 301 420 288      -5.91
cases  1960-12-25 1961-01-01 one_step 1960-12-25 7    trained  fixed_point 230 322 451 340      -6.02
cases  1961-01-01 1961-01-08 one_step 1961-01-01 7    held_out fixed_point 247 349 488 402      -6.37
```

The values are illustrative. The manifest of that run says: "parameters: a fixed
point from he2010.toml; no parameter uncertainty. Training rows: declared by
--trained-through, not verified."

Removed spellings fail with a message naming the replacement and are not kept as
aliases: `camdl fit predict`, the `horizon` column holding a mode, and (after
the last stage) `pfilter --save-prequential`, `--score-from` and
`--save-filtered-streams` (D103).

## What this proposal does not do

- It does not refit at each origin. Parameters are fixed from one fit (D35).
- It does not filter once and fork at each origin; each origin repeats the
  filter over the rows before it (D102). The fork is an optimisation for a
  later, measured change to `particle_filter.rs`.
- It does not make `simulate --init-state` exact inside a period (PR #1158).
- It does not import or export Hubverse files, or score forecasts made outside
  camdl (D37).
- It does not add aggregated targets (weekly targets from a daily stream), the
  calendar `covers` forms (D45), or data vintages.
- It does not design the per-forcing hold (gh#1300) or the covariate note
  (gh#1083).
- It does not change `particle_filter.rs`, the saved-state format, or any
  likelihood.

## Stages

Each stage is one pull request with its tests and a `Proposal:` footer, and each
leaves `main` consistent.

1. **Finish Stage 5.** Delete the free-forward tail (`forecast_times`,
   `leaf_row_coverages`; gh#1151, gh#1152): free-forward predicts the bound rows
   and nothing past them. Rename the `horizon` column that holds a mode to
   `mode`. Update the spec's "Not yet built (Phase B)" markers for what Stage 5
   built.
2. **The row and the table.** `PredictionRow`, `Quantity`, `RowRole`;
   `predictions.tsv` and `prediction.json` written by all three modes;
   `convergence.tsv`; the shared reader (gh#1259, gh#1260). No new capability.
3. **Parameter sources.** `ParameterSource`, `ParameterSet`; `--params`,
   `--trained-through`; single-estimate fits accepted and labelled (D101). The
   command becomes `camdl predict` (D100).
4. **Origins.** `StateConditioning` on the plan; `--origin`, `--origins`;
   several plans per run (D102).
5. **Forecast scores (the runbook's Stage 6).** `ScoreArtifact` with
   `ScoreBody::Forecast`, `ArtifactKind::Score`, `camdl score --mode forecast`,
   `scored.tsv`, `summary.tsv`. Proposal A's steps A6 and A7, amended as above.
6. **One-step scores through the same command.** `camdl score --mode one-step`,
   in process (D104); `compare` reads stored scores (A8). The fit-identity step
   (A4) lands here, as its own `HASH_VERSION` bump, after confirmation.
7. **Trim `pfilter`.** Remove its three predictive outputs once stage 6 gives
   the same numbers (D103).

Stage 1 needs no decision beyond approval of this proposal. Stage 2 needs D105,
stage 3 D100 and D101, stage 4 D102, stage 6 D104, stage 7 D103.

## Tests

Each is written to fail before its stage.

- **One table, three modes.** A fit with a mid-series `holdout_after`: every row
  of `predictions.tsv` has all key columns; one-step rows after the cutoff are
  `held_out`; forecast targets are `future`; the free-forward file has no row
  past the data.
- **Quantity.** For a Poisson stream at a fixed point, the expectation band's
  variance plus the mean of the expectation equals the observation band's
  variance within Monte Carlo error (the law of total variance); for a model
  with no observation noise term the two bands coincide.
- **Fixed point equals a one-draw posterior.** `predict --params P` and a
  posterior whose every draw is `P` give identical `predictions.tsv` apart from
  the `parameters` column.
- **Fixed point is labelled.** The manifest and the table say `fixed_point`;
  with no `--trained-through`, every role is `unknown` and `score` stamps
  nothing held-out; with it, the stamp is "held-out as declared".
- **One-step score equals the filter's likelihood.** At a fixed point and
  matched seed, the sum of `scored.tsv`'s one-step `log_score` over all rows
  equals `pfilter`'s log likelihood exactly.
- **An origin later than the training extent.** The spec §4.4 reference model
  (daily and weekly incidence): with the origin two weeks past the training
  extent, the predictive mean and variance of the weekly flow agree with an
  importance-sampling reference that conditions on the daily rows through the
  new origin. The same model with the origin at the training extent reproduces
  today's forecast bit for bit.
- **No leak per origin.** For each origin, changing any row ending after it
  leaves that origin's predictions bit-identical; changing a row ending before
  it changes them.
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
  Software_ 69(12). doi:10.18637/jss.v069.i12. The paper was not re-read for
  this proposal; the statements about pomp rest on the package manual sources,
  version 6.5.0.1 (`man/pfilter.Rd`, `simulate.Rd`, `forecast.Rd`,
  `cond_logLik.Rd`, `pred_mean.Rd`, `R/forecast.R`):
  <https://github.com/kingaa/pomp>.
- King AA, Ionides EL, Wheeler J. Simulation-based Inference for Epidemiological
  Dynamics, lesson 7, "Case study: Ebola, model diagnostics, and forecasting":
  <https://kingaa.github.io/sbied/ebola/>.
