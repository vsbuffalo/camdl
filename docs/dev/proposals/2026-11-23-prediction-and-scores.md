# Prediction and scores: one vocabulary, `camdl predict` and `camdl score`

Status: implemented (65bd81771, a0daab8bf, e0f8457a9, eaa99988a).

Builds on: `2026-11-20-run-store-api.md` (the run store, on main), whose
"Scores" section defines the score's identity and leaves its command to this
document; `docs/engineering/forecasting.md` (the temporal spec), whose terms
this document uses. Supersedes: the draft `2026-11-19-unified-prediction.md`
(never merged), for everything it says about scores and the vocabulary of
prediction. Its sections on forecasts from several origins and on covariate
statements are deferred, as listed under "Deferred". Amends:
`2026-11-14-score-artifact.md` and Stage 6 of
`2026-11-15-temporal-architecture-migration.md`, as listed under "What changes
in the earlier proposals".

Claims about current behaviour are marked _verified_ with the file and line or
the command that checked them, or _inferred_.

## Summary

camdl can predict from a stored fit or from a fixed parameter point, in three
modes, and stores every prediction. It cannot yet say how good a prediction was.
This proposal adds that, as a second command. `camdl score` takes a prediction,
binds the observed outcomes of the rows the prediction was not allowed to see,
and stores a score: per row, the log predictive density of the observed value,
its continuous ranked probability score, its weighted interval score and its
position in the predictive distribution; per stream, mode and lead, their means
and the coverage of the central intervals. The outcomes change only the score,
never the prediction. `camdl compare` reads stored scores instead of computing
its own and discarding them. Every prediction table gains a `quantity` column
saying whether its bands are of observations or of their expected value, and a
`mode` column in place of the `horizon` column that holds a mode name today. A
forecast that reads a table-driven forcing after its origin is refused until the
user states what those values are.

The first part of the document, "Concepts", defines the words that the rest uses
and that the user documentation will use. It is written to move into `docs/`
unchanged.

## Concepts

This section is the vocabulary of prediction in camdl. Each term below is used
in exactly this sense in the commands, their output and their documentation.

### Rows, streams and the time a row is scored

A model's `observations {}` block declares **streams**: each is one observed
series, such as weekly reported cases. A **row** of a stream's data is one
observation. Most rows measure a **period**, a stretch of time over which
something was counted (the cases reported in the week ending 7 January), and
some measure an instant (a prevalence survey on one day). The **end** of a row
is the end of its period, or its instant. A row is predicted and scored at its
end, because that is when its value is complete.

### Training rows, held-out rows and conditioning rows

Any prediction camdl makes is the distribution of a row `y` given parameters `θ`
and a latent state `x`, the model's compartments at that time. Two sets of rows
describe what that prediction was allowed to see, and they are different sets.

- The **training rows** are the rows `θ` was fitted to. The latest end among
  them is the **training extent**. Every other row is a **held-out row**: the
  parameters never saw its value. "In-sample" and "held-out" describe a row
  against the training rows.
- The **conditioning rows** of a prediction are the rows its latent state `x`
  had seen when the row was predicted. A particle filter carries a swarm of
  possible states forward in time. When it reaches a row it **assimilates** it:
  it weights each particle by how likely that particle makes the row's value and
  resamples, so from then on the swarm reflects the row. A row that has not been
  assimilated has not entered the state at all. The conditioning rows are the
  rows assimilated before the row being predicted.

The two sets answer different questions, and they overlap differently in each
mode. In a fit the latent state is a set of random variables alongside `θ`, and
a sampler such as PMMH or PGAS infers both from the training rows. A prediction
does not reuse the latent state the fit inferred. For each parameter draw it
runs the model again, and the latent state at each predicted row is estimated
only from that prediction's conditioning rows. So the parameters can have seen a
row that the latent state has not, and the reverse:

- In a one-step prediction of training row 5, `θ` was fitted to rows 1 to 7 and
  the latent state has seen rows 1 to 4. The prediction is in-sample in `θ` and
  honest in the state.
- In a one-step prediction of held-out row 9, `θ` was fitted to rows 1 to 7 and
  the latent state has seen rows 1 to 8, two of them held out. The parameters
  are not refitted, so the row is held out in `θ`; the state has assimilated the
  held-out rows before it, as a forecaster running the model each week would.
- In a forecast from the training extent the two sets coincide: rows 1 to 7.
- In a free-forward prediction the conditioning rows are empty.

Roles (below) are stated against the training rows, because they say what the
parameters had seen. The mode says what the latent state had seen.

The three modes of prediction are three choices of conditioning rows.

Take one weekly stream with twelve rows, the first seven of them training rows.
`■` is a row the filter has assimilated when the prediction is made, and `○` is
a row being predicted.

```
row ends        1   2   3   4   5   6   7   8   9   10  11  12
              ──┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴──▶ time
                └──── training rows ────┘   └─── held-out rows ──┘

free_forward    ○   ○   ○   ○   ○   ○   ○   ○   ○   ○   ○   ○
                no conditioning rows: the model run forward from its start

one_step        ■   ■   ■   ■   ○                          predicting row 5
                ■   ■   ■   ■   ■   ■   ■   ■   ○          predicting row 9
                each row predicted from every earlier row, then assimilated

forecast        ■   ■   ■   ■   ■   ■   ■ ┃ ○   ○   ○   ○  origin after row 7
                the filter stops at the origin; no row after it is
                assimilated
```

- **Free-forward** prediction asks whether the model, at these parameters,
  generates data like the observed series. No filter runs, and every backend can
  do it.
- **One-step** prediction predicts each row from every row before it. It needs a
  particle filter, and so a stochastic model: the filter estimates the latent
  state from the earlier rows, and the prediction is made from that estimate.
  camdl's filter runs on the chain-binomial backend. Its lead is always one
  row's period, and one filter pass gives every row's prediction. On the ODE
  backend the state at given `θ` is fixed by the equations and has nothing to
  learn from earlier rows, so a one-step prediction would equal the free-forward
  one.
- **Forecast** prediction assimilates the rows up to the **forecast origin** and
  predicts past it with nothing assimilated in between, so its lead grows with
  each row. It too needs a particle filter.

`camdl simulate` reads no data at all and is outside these modes: it generates
from the model.

### Origin, target, lead and horizon

A forecast's **origin** is the instant after which the filter assimilates no
row. In camdl today it is the training extent of a fit, or `--holdout-after` for
a fixed point. A **target** is one period of one stream that a forecast
predicts; the targets past the origin come from each stream's declared rhythm
(its data's period, or its `emit_schedule`), since there may be no rows there. A
stream whose data gives each row's window in columns and that declares no
`emit_schedule` has no rhythm to continue, and takes its targets from the
held-out rows' dates instead.

A target's **lead** is the time from the origin to the target's end. Its
**horizon** is its position among its stream's targets, counting the first one
that ends after the origin as 0. The command-line flag `--horizon "4 weeks"`
gives how far past the origin the targets extend, which this document calls the
**reach**; the flag keeps its name, and the output's `horizon` column holds the
integer position. In `forecast/<stream>.tsv` it already does, beside a `lead`
column; in `predictive/<stream>.tsv` today a column of that name holds the mode.

One forecast's targets at different leads are a joint draw: with no resampling
after the origin, each sample is one trajectory through every target. One-step
predictions at different rows are not jointly distributed in any usable way.

### Where the parameters came from

A prediction's **parameter source** is one of two.

- A **posterior**: the draws of a stored sampler fit (PMMH, PGAS, or MH on the
  ODE backend), written `camdl predict @run`. Each predictive sample comes from
  one draw, so the bands carry parameter uncertainty.
- A **fixed point**: one parameter vector given on the command line, written
  `camdl predict model.camdl --params p.toml`. The point is repeated, each copy
  under its own seed, so the bands carry the model's process noise and its
  observation noise and no parameter uncertainty. Every output from a fixed
  point is labelled `plug_in`.

For a fixed point, `--holdout-after DATE` declares that the point was fitted to
the rows ending on or before `DATE`, and the prediction assimilates no row after
it: `predict` reads nothing past `DATE`. camdl records the declaration (the
identity's `trained_through`) and labels the rows up to it `trained`, but cannot
verify it. A stored fit's training rows are checked against the data by their
digests; a fixed point's are the user's word.

Not every backend predicts in every mode. One-step and forecast prediction need
a particle filter, which camdl runs on the chain-binomial backend only:

| Backend        | `free_forward` | `one_step` | `forecast` |
| -------------- | -------------- | ---------- | ---------- |
| chain-binomial | yes            | yes        | yes        |
| Gillespie      | yes            | no         | no         |
| ODE            | yes            | no         | no         |

### What a band shows

For one row there are two different random quantities, and a band can be of
either.

- The **observation**, `y`: a draw from the observation model. Its spread
  carries the uncertainty about the parameters and the state and the observation
  noise (a negative-binomial count's scatter about its mean). Observed data are
  compared with this one: about 90% of observed rows should fall inside its 90%
  band, and every score is computed against it.
- The **expectation**, `E[y | x, θ]`: the observation model's mean at one state
  and one parameter vector, with no observation noise drawn. It is a
  distribution too, over particles and draws, and its band is what is usually
  drawn as "the fitted curve with its uncertainty". Observed data are not
  expected to fall inside it.

The two have the same mean. By the law of total variance, the observation's
variance is the expectation's variance plus the mean of the observation model's
own variance, so the difference in width between the two bands is the part of
the spread that is observation noise. Laying data over an expectation band makes
a model look overconfident when it is not.

### What a score is

A **score** compares a prediction with the outcome once the outcome is known. A
**scoring rule** is the function that turns a predictive distribution and an
observed value into one number; each rule measures something different, and a
score holds several. For one scored row camdl computes:

- the **log score**, the log of the predictive density at the observed value.
  Higher is better. Summed over rows, it is the quantity whose differences
  between models are compared (Gneiting & Raftery 2007, §4.1).
- the **CRPS** (continuous ranked probability score), the expected absolute
  distance between a predictive draw and the observed value, minus half the
  expected distance between two independent draws. Lower is better, and it is in
  the units of the data (Gneiting & Raftery 2007, §4.2). camdl estimates it from
  the predictive samples with the fair estimator, which removes the bias a
  finite sample gives the second term (Ferro 2014).
- the **WIS** (weighted interval score), computed from the predictive quantiles:
  a weighted sum, over the central intervals, of each interval's width plus a
  penalty for an observation that falls outside it, and the absolute error of
  the median. Lower is better. It approximates the CRPS, and splits into three
  parts that are reported beside it: dispersion (the widths), and the penalties
  for observations above and below the intervals (Bracher et al. 2021).
- the **PIT** (probability integral transform), the predictive distribution
  function at the observed value, randomized within the jump for a count. Over
  many rows of a calibrated prediction it is uniform.
- whether the observed value fell inside the central 50% and 90% intervals.

The WIS is the rule the public epidemic forecast hubs score by. The US COVID-19
Forecast Hub and the CDC's FluSight challenge since 2021 collect forecasts as a
fixed set of quantiles and evaluate them by WIS and interval coverage (Cramer et
al. 2022; Bracher et al. 2021). A quantile forecast has no density, so these
hubs do not use the log score; FluSight's earlier seasons scored binned
probabilities with a log score (Reich et al. 2019). camdl stores every
predictive sample, so it computes both, and reports the WIS on the quantile set
the hubs use, so a camdl forecast's WIS can be read beside a hub's. Exporting
forecasts to a hub's file format is outside this proposal.

Every rule is computed per stream: each row of a stream is scored against that
stream's own predictive distribution, and every summary is per stream. A
stream's scores are never added to another stream's except in one place: under
one-step and forecast prediction the log score of all the streams observed at
one time, from their joint predictive density, is reported beside the per-stream
ones as `joint`.

A score is a stored result of its own, separate from the prediction it scores,
for three reasons. The outcome must not be able to change the prediction: a
forecast made today and scored next month is the same stored prediction both
times. A score means something only with the prediction's parameter source, mode
and lead attached, so it carries them. And scores are compared across models
many times, so they are computed once and kept.

Every predicted or scored row has a **role** against the training rows:
`trained` (the parameters were fitted to it), `held_out` (they were not, and the
row's value was read), `future` (a period the prediction read no row for, which
includes every forecast target), or `unknown` (a fixed point with no
`--holdout-after`). A fixed point's rows after its `--holdout-after` are
`held_out` on the user's declaration, which camdl cannot check, so every score
of them is marked "held out as declared, not verified" and `compare` does not
difference such a score against one whose training rows were verified. Scores of
different roles, different modes or different leads are never summed or averaged
together. A mean log score at one week's lead and one at eight weeks' lead are
different measurements.

### How the predictive density is estimated

The predictive density of a row is estimated from the samples the prediction
drew. With `M` parameter draws and `N` particles per draw, and `ℓ_{m,i}` the
observation model's log density of the observed value at particle `i` of draw
`m`, the log score is

```
log p̂(y) = logsumexp_{m,i} ℓ_{m,i} − log(M·N)
```

This is the log of the average density over every sample. It is never the
average of per-draw log scores, which by Jensen's inequality is a smaller and
different number. For a fixed point `M = 1`. Under one-step prediction at a
fixed point, this per-row quantity is the conditional log-likelihood pomp calls
`cond_logLik` (pomp 6.4; King, Nguyen & Ionides 2016), and its sum over rows is
the particle filter's log-likelihood estimate.

The **basis** of a score is how the state was represented. Under a particle
filter (the chain-binomial backend) the state is a particle swarm, and `p̂(y)` is
a Monte Carlo estimate whose log is biased downward, by more at fewer particles.
On the ODE backend the state at given `θ` is the deterministic trajectory, so
`N = 1` and `ℓ_m` is the exact observation density along draw `m`'s trajectory.
Scores on different bases differ partly by the estimator, and are not
differenced against each other.

## What exists today

_Verified_ on main at `bbee1bb13` unless marked.

- **Predictions, both parameter sources.** `camdl predict @run` and
  `camdl predict model.camdl --params …`, modes `free_forward`, `one_step`,
  `forecast`, stored under the run store.
- **Prediction tables.** `predictive/<stream>.tsv` with a `horizon` column that
  holds the mode name (`fit/predict.rs:53-58`) and a `role` column (`trained`,
  `held_out`, `future`; `predict.rs:62-90`); `forecast/<stream>.tsv`, already
  with an integer `horizon` and a `lead` (`forecast_output.rs:684`);
  `forecast-samples.tsv`, keeping 5 particles per draw
  (`forecast_output.rs:33`). No column says which quantity a band shows.
- **One-step prediction from a stored fit.** Filters through the stored
  problem's lines, which are the training rows only
  (`bindings.training_by_source()`, `predict.rs:5019`), so its bands end at the
  training extent and no row has role `held_out` (_inferred_ from the code; not
  run).
- **One-step scores at a fixed point.**
  `pfilter --params … --save-prequential DIR`: a `PrequentialTrace`
  (`score/src/trace.rs:285`) with the log score, CRPS and PIT per row. Rows
  before `--score-from` are dropped from it.
- **The expectation band.** `pfilter --save-filtered-streams`, `phase`
  `predictive` and `filtered`, unlabelled as an expectation.
- **Forecast log scores.** The driver records a target's per-particle log
  density when the target's held-out value is supplied (`forecast_driver.rs:61`,
  `:344`); `predict` supplies none, and `marginal_log_scores`
  (`fit/forecast_score.rs:80`) is called only by tests. The driver can also
  record each particle's expected value (`record_expected`), which no command
  asks for.
- **Held-out scores of a stored fit.** Refused by `compare`
  (`refuse_held_out_scoring`, `compare.rs:2628`): "a stored fit keeps those
  rows' dates, not their values, so compare cannot score them".
- **In-sample scores in `compare`.** Computed by `camdl pfilter --no-store`
  child processes in a temporary directory and discarded (`compare.rs:2644`).
- **Score labels.** `score::Provenance` (`PlugIn`, `Posterior { n_draws }`),
  `PredictiveBasis` (`Filtered`, `OdeFreeForward`), `Conditioning` (`InSample`,
  `HoldOutTail`) in `score/src/trace.rs:48-125`.
- **Score kernels.** `crps_sample_fair`, `pit_sample_randomized`
  (`score/src/trace.rs:820`, `:872`).
- **A forecast reading a forcing after its origin.** Read as it is, with no
  note, while the table covers the reach; temporal spec §4.6 lists the note as
  not built. A table that ends before the reach is refused by default
  (`forcing-outside-range`, `forcing_extrapolation.rs`), and held at its last
  value with a warning under `--allow-forcing-extrapolation` (_read_, not run).
- **Design values after the origin.** Stated already: `--design` and
  `DesignMode::{Prospective, ExPost}` (`forecast_plan.rs:49-60`);
  `--realised-design` is refused until held-out values are available.

## The commands

Each command has one job, and reads observed values only as its job needs.

- **`simulate`** generates from the model. It reads no observed values.
- **`pfilter`** computes the likelihood at one parameter vector, with that
  filter's diagnostics. It reads every bound row.
- **`predict`** makes predictions: bands and samples, each recorded with its
  training and conditioning rows. It reads only its conditioning rows.
- **`score`** scores a stored prediction against the outcomes, and writes the
  table that lays the two side by side. It reads the outcome of every scored
  row.
- **`compare`** differences the stored scores of several models. It reads no
  observed values: it reads scores.

### `camdl score`

```
camdl score <PREDICTION> --data STREAM=FILE ... [--store DIR] [--label L] [--force] [-o DIR]
```

`score` takes a stored prediction's handle and the outcome files, and nothing
else. Everything about the prediction (its parameter source, mode, reach,
particles, seed, scenarios) is already in the prediction's identity, so `score`
has no flags for it. The `--data` files hold the outcomes. For a stored fit they
are the files the fit was given, held-out rows included; for a fixed point they
are the files the prediction was given, extended past `--holdout-after`.

A score takes two commands, one to predict and one to score:

```bash
camdl predict @sir --mode forecast --horizon "8 weeks" --label sir_fc
camdl score @sir_fc --data cases=london.tsv
```

Scoring a prediction from `predict`'s own arguments in one command is left out.
It would make `score` repeat every flag and refusal of `predict` for a saving of
one line, and a later version can add it without changing what is stored.
Free-forward predictions are scored too: the scored rows are every bound row,
with their roles.

Before scoring, `score` checks that the outcome files agree with what the
prediction saw: every conditioning row of the prediction must be present in the
outcome file and equal, by row digest. A file that disagrees is refused, naming
the stream and the first row that differs, since its held-out rows would then
belong to a different series from the one the prediction conditioned on.

### What `score` computes, by mode

- **Free-forward.** Each scored row's density is the observation density at each
  sample's state at that row. This is the only mode on the ODE backend, where it
  gives each posterior draw's exact density along its trajectory, and it is how
  a stored MH fit on the ODE backend gets held-out scores.
- **One-step.** One-step prediction of a held-out row needs the rows before it
  assimilated, and those include earlier held-out rows. So `score` runs the
  one-step filter onward from the prediction's last conditioning row through the
  held-out rows, predicting each row from the swarm before it, recording `ℓ`,
  and then assimilating it. The pass that predicts a held-out row is the pass
  that scores it. Its one-step bands for the held-out rows are stored in the
  score, never in the prediction, because they depend on held-out values. This
  matches what `predict` does today: a one-step prediction from a stored fit
  filters the stored training rows only, and ends at the training extent.
- **Forecast.** After the origin nothing is assimilated, so each target's `ℓ` is
  the observation density at each sample's state at that target. The driver
  records it per particle when it is handed the target's observed value, and the
  weights are equal by construction. The joint log score of one forecast sums
  `ℓ` over a sample's targets before the average over samples.

In every mode `score` recomputes the prediction's samples rather than reading
them from the stored prediction, since the stored files keep summaries and, for
a forecast, 5 particles per draw, and the density needs every particle. It runs
the prediction's own computation, with each draw's own seed and the outcomes
bound, and checks that the predictive bands it recomputes equal the stored
prediction's byte for byte before it writes a score. Binding an outcome changes
no weight and draws no random number, so the two agree; a disagreement is
refused as an internal error. A score therefore costs about what its prediction
cost.

### Output

The score's layout is amended by `2026-11-27-score-layouts.md`.

A score is stored inside the prediction it scores (run-store proposal,
"Scores"), and holds:

- `scored.tsv`: one row per scored row or target, laid out for plotting.
  ```
  scenario stream <dims> t_start t_stop [date_start date_stop] mode origin
  horizon lead role parameters quantity mean q05 q25 q50 q75 q95
  observed log_score crps wis wis_dispersion wis_over wis_under pit in_50 in_90
  ```
- `summary.tsv`: one row per (stream, mode, role, lead).
  ```
  stream <dims> mode role lead n log_score_sum log_score_mean crps_mean
  wis_mean coverage_50 coverage_90
  ```
  A `stream` value of `joint` marks the log score of all the streams observed at
  one time; its other rule columns are empty.
- `score.json`: the parameter source, the basis, the conditioning, the reach,
  the particle count, the scored rows' digests, and the notes.

`camdl show` and `camdl cat` read the score. `camdl compare` reads stored scores
(below).

### Before and after

```bash
# a fit on 1950-1960, 1961-1964 held out by holdout_after
# before: no held-out score; compare refuses
camdl compare @sir @seir

# after: predict, score, then compare the stored scores
camdl predict @sir  --mode one_step --label sir_os
camdl predict @seir --mode one_step --label seir_os
camdl score @sir_os  --data cases=london.tsv
camdl score @seir_os --data cases=london.tsv
camdl compare @sir @seir --mode one_step      # finds the two stored scores
```

```bash
# a published parameter vector, scored row by row over the whole series
# before: pfilter, dropping the rows before --score-from
camdl pfilter measles.camdl --params he2010.toml --data cases=london.tsv \
    --particles 2000 --save-prequential he --score-from 1961-01-01

# after: every row, each with its role
camdl predict measles.camdl --params he2010.toml --data cases=london.tsv \
    --mode one_step --particles 2000 --label he
camdl score @he --data cases=london.tsv
```

`scored.tsv` from the second, abridged (values illustrative):

```
stream t_start    t_stop     mode     lead role    parameters quantity    q05 q50 q95 observed log_score
cases  1960-12-18 1960-12-25 one_step 7    unknown fixed_point observation 212 301 420 288      -5.91
cases  1960-12-25 1961-01-01 one_step 7    unknown fixed_point observation 230 322 451 340      -6.02
```

## Labelling the predictions

### The `quantity` and `mode` columns

Every prediction table gains a `quantity` column. Every band `predict` writes
today is of observations, so the column holds `observation` on every row; the
column is written so that no band can be read as the other kind. The
`predictive/<stream>.tsv` `horizon` column, which holds the mode name today, is
renamed `mode`, and `horizon` becomes the integer position defined above, empty
outside a forecast. `forecast/<stream>.tsv` keeps its integer `horizon` and its
`lead`, and gains `mode` and `quantity`. `pfilter --save-filtered-streams` gains
the same `quantity` column, holding `expectation`.

An expectation band from `predict` (`--quantity expectation`) is deferred. For a
forecast the driver can already record it (`record_expected`). For one-step it
needs each particle's observation-model mean recorded beside its draw in the
filter's one-step record, which is a change to `particle_filter.rs`, a high-risk
file, for a band `pfilter` already writes.

### The role column

`role` in a prediction table already takes `trained`, `held_out` and `future`.
This proposal adds `unknown`, for the rows of a fixed point given no
`--holdout-after`, which today are labelled `trained`. A prediction table writes
`held_out` only for a row whose value it read; since neither parameter source
reads held-out values on the store, held-out rows appear with their values in
scores.

## A forecast that reads a forcing after its origin

A forcing read from a data table (births, population) is an input to the model
at every time. After a forecast origin its values are one of three things that
camdl cannot tell apart: fixed or assumed in advance (a planned campaign, a
scenario), observed later, or missing. A forecast made with values observed
after its origin is an after-the-fact evaluation, and its scores flatter the
model.

A forecast whose reach reads a table-driven forcing after the origin is refused,
before any filter runs, naming the forcing, the date its table ends, the origin
and the reach, until the user states which it is:

```
error: forcing `pop` is read after the forecast origin and nothing says how.
  origin 1961-01-01, reach 8 weeks (to 1961-02-26); its table `pop` has rows to 1964-12-31.
  State one of:
    --covariate pop=prospective   the values after the origin were fixed or assumed at the origin
    --covariate pop=realised      they were observed after the origin
    --covariate pop=hold          carry the value at the origin forward
```

- `prospective` reads the table as it is; the forecast is conditional on those
  values, and its manifest names the table and its digest.
- `realised` reads the table as it is, and labels the prediction and every score
  of it "conditional on covariates observed after the origin". `compare` refuses
  to difference such a score against one not so labelled.
- `hold` cuts the table at the origin and carries its last value forward. It is
  refused for a cubic-spline table, because cutting a natural spline's knots
  changes the forcing before the origin too.

"Reads after the origin" is judged on the table's argument: for a forcing with a
`lag`, `t − lag`, so a lagged forcing needs a statement only once the reach
exceeds the lag. The statement enters the prediction's identity. Forcings that
need none: a periodic kind, a formula in `t`, and a `piecewise` forcing, which
holds its last segment by declaration (a note names the breakpoint when the
reach passes it). An `interpolated` table, `method = constant` included, is a
table and needs a statement. A table that ends before the reach stays refused as
today (`forcing-outside-range`), and `--allow-forcing-extrapolation` keeps
holding its last value with a warning; `hold` is the same treatment stated for
one forcing and recorded in the identity. An estimated `bspline` past its window
stays refused as today.

The vocabulary follows the design values a forecast already states
(`DesignMode::Prospective` and `ExPost`, `--design` and `--realised-design`):
`prospective` is the same word, and `realised` is the flag's word for `ExPost`.

## Types

The per-row `lead` and the score's rows are amended by
`2026-11-27-score-layouts.md`.

The score's identity is the one the run-store proposal defines, with its fields
filled in:

```rust
#[derive(Identity, Clone)]
#[identity(tag = "score", version = 1)]
pub struct ScoreIdentity {
    /// The prediction scored; its identity holds the parameter source, the
    /// mode, the reach and everything the prediction read.
    pub prediction: Dep<Prediction>,
    /// Per stream, the outcome rows scored: the lines of the bound file after
    /// the prediction's conditioning rows, as a stored data table.
    pub outcomes: BTreeMap<String, Dep<DataTable>>,
    /// The interval levels and the PIT seed; the rules are fixed by the
    /// score's version.
    pub settings: ScoreSettings,
}

pub(crate) struct ScoreSettings {
    pub intervals: [Finite; 2],   // 0.5, 0.9
    pub pit_seed: u64,
}
```

A score computes every scoring rule at once, so there is one score per
prediction and set of outcomes, stored at `scores/score-<hex>/` inside the
prediction. The rules share their inputs, the same samples and, for one-step,
the same filter pass, which is most of a score's cost, and `compare` reads them
together. A rule added later bumps the score's version, which recomputes the
stored scores at about the cost of the predictions they rerun. The run-store
proposal's `rule` field and its `scores/<rule>-<hex>` layout are replaced
accordingly.

The row role and the quantity are typed:

```rust
pub enum RowRole { Trained, HeldOut, Future, Unknown }
pub enum Quantity { Observation, Expectation }
```

The existing labels in the `score` crate are kept and extended:

| Concept          | Type                             | Change                                                                                                                                                                                        |
| ---------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| parameter source | `score::Provenance`              | none: `Posterior { n_draws }` and `PlugIn` map onto the two sources                                                                                                                           |
| basis            | `score::PredictiveBasis`         | none                                                                                                                                                                                          |
| training rows    | `score::Conditioning`            | adds `Unknown` (a fixed point with no `--holdout-after`) and `HoldOutDeclared` (a fixed point's rows after `--holdout-after`); a fixed-point score is no longer stamped `InSample` by default |
| mode and lead    | a `mode` and a per-row `lead`    | new on the trace                                                                                                                                                                              |
| covariates       | `CovariateStatement` per forcing | new, in the prediction's identity                                                                                                                                                             |

## `compare` reads stored scores

The score files `compare` reads are amended by `2026-11-27-score-layouts.md`.

`compare` keeps its command line and its table, and changes where its numbers
come from. Today it runs `camdl pfilter` child processes for each fit, scores
the training rows in a temporary directory, and discards the result; it refuses
held-out scoring of a stored fit (`compare.rs:2628`). Its table today has the
columns `Model`, `T_score`, `elpd`, `Δelpd`, `LR`, `se(Δ)`, `evidence`, and
`crps`, `Δcrps` and `PIT_cov90` when a single stream was scored
(`compare.rs:3835-3840`), with a per-stream block below it otherwise.

After this change `compare` finds each fit's stored score for the requested
mode, and computes one through `predict` and `score` when none is stored. It
then differences stored numbers. So held-out scores of stored fits work, and the
table gains what a score holds that it did not:

- a `role` column, so held-out and in-sample rows are never shown as one number,
  and a `lead` column for forecast scores, one table row per (model, role,
  lead);
- `wis` and `Δwis` beside `crps`, and `cov50` beside `PIT_cov90`;
- `--mode one_step|free_forward|forecast`, defaulting to one-step where the
  backend has it and free-forward otherwise, as today.

It refuses to difference two scores whose mode, reach, scored rows, role, basis
or covariate statements differ, naming the field; its existing checks
(`score::preflight`) are these refusals over the trace's fields. The child
`pfilter` processes and their temporary directory go. `--in-sample` keeps its
meaning: it scores the training rows, whose role is `trained`.

## Deferred

Each of these is a later proposal of its own, written when someone needs it.

- **Forecasts from many origins.** Moving the origin through the held-out rows,
  with the parameters of one fit, gives many forecasts at each lead and so a
  mean score per lead. It needs `ForecastPlan::new` to take the origin and the
  conditioning rows as arguments and one driver call per origin. The
  unified-prediction draft's "Many origins" and "Origins" sections are its
  starting point.
- **The rest of the covariate design.** A `[forecast.covariates]` key in the fit
  file, and `predict --table` supplying a table longer than the fit's.
- **One prediction table.** Replacing `predictive/`, `forecast/` and their
  manifests with one `predictions.tsv` and one `prediction.json`, and moving the
  convergence columns to a file of their own.
- **The expectation band from `predict`**, as above.
- **Removing `pfilter`'s predictive outputs** (`--save-prequential`,
  `--score-from`, the `predictive` rows of `--save-filtered-streams`) once
  `score` gives the same numbers.

## What changes in the earlier proposals

- **The run-store proposal**, "Scores": the identity's `holdout` field is named
  `outcomes`, its `rule` field is replaced by `settings`, and the layout is
  `scores/score-<hex>/`.
- **`2026-11-14-score-artifact.md`**: its trace-file and `--save-pass` steps are
  replaced by in-process scoring, its `ScoredLeadTime` by the `mode` and `lead`
  fields, and its identity sections by the run store's.
- **Temporal-architecture migration, Stage 6**: forecast scoring is
  `camdl score
  --mode forecast`, with the log score as defined here (route 2,
  the held-out target's density at each particle).

## Stages

Each stage lands on its own with its tests.

1. **Labels.** The `quantity` column on every prediction table and on
   `filtered_streams.tsv`; `mode` in place of the mode-valued `horizon` in
   `predictive/`; the `unknown` role for a fixed point with no
   `--holdout-after`. The prediction byte pins move, in their own commit.
2. **The covariate refusal and statements.** `--covariate`, the plan-time
   classification and refusal, the statement in the prediction's identity.
3. **Forecast scores.** `ScoreIdentity`, the score artifact, `camdl score` for a
   forecast from both sources, `scored.tsv`, `summary.tsv`, `score.json`.
4. **One-step and free-forward scores.** The onward one-step pass through the
   held-out rows; free-forward scores, on the ODE basis included; the new
   `Conditioning` variants.
5. **`compare` on stored scores.** The child processes and the held-out refusal
   go.

## Tests

Each is written to fail before its stage.

- **The pfilter tie.** At a fixed point on a chain-binomial model, the sum of a
  one-step score's per-row log scores over every row equals `pfilter`'s
  log-likelihood at the same particles and seed.
- **The ODE density.** On the ODE backend at a fixed point, each row's log score
  equals the observation density at the trajectory, computed independently from
  `simulate`'s trajectory and the observation model.
- **No leak.** Editing a held-out outcome changes the score's run id and not the
  prediction's, and the prediction's stored bytes are identical with and without
  a score.
- **The WIS.** On a predictive distribution with known quantiles, the WIS and
  its three parts equal a hand computation by the formula of Bracher et al.
  (2021).
- **The mixture.** On a posterior of two draws with known per-draw densities,
  the log score is the log of their average, and differs from the average of
  their logs.
- **Never pooled.** `summary.tsv` has one row per (stream, mode, role, lead) and
  `compare` refuses scores at different leads, roles or bases, naming the field.
- **Outcome files agree with the prediction.** An outcome file that differs from
  the prediction's conditioning rows in one row is refused, naming the row.
- **The covariate refusal.** A forecast reading a table forcing after its origin
  is refused with the dates; each statement then runs, and `realised` labels the
  score.
- **Labels.** Every prediction table has `quantity` and `mode`; a fixed point
  with no `--holdout-after` labels its rows `unknown`.
- **The score recomputes the prediction.** A score's recomputed bands equal the
  stored prediction's byte for byte, for both sources and every mode a backend
  has.

## References

- Bracher, J., Ray, E. L., Gneiting, T. & Reich, N. G. (2021). Evaluating
  epidemic forecasts in an interval format. _PLOS Computational Biology_ 17(2),
  e1008618. doi:10.1371/journal.pcbi.1008618
- Cramer, E. Y. et al. (2022). Evaluation of individual and ensemble
  probabilistic forecasts of COVID-19 mortality in the United States.
  _Proceedings of the National Academy of Sciences_ 119(15), e2113561119.
  doi:10.1073/pnas.2113561119
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. _Quarterly Journal
  of the Royal Meteorological Society_ 140(683), 1917–1923. doi:10.1002/qj.2270
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. _Journal of the American Statistical Association_
  102(477), 359–378. doi:10.1198/016214506000001437
- King, A. A., Nguyen, D. & Ionides, E. L. (2016). Statistical inference for
  partially observed Markov processes via the R package pomp. _Journal of
  Statistical Software_ 69(12). doi:10.18637/jss.v069.i12
- Reich, N. G. et al. (2019). A collaborative multiyear, multimodel assessment
  of seasonal influenza forecasting in the United States. _Proceedings of the
  National Academy of Sciences_ 116(8), 3146–3154. doi:10.1073/pnas.1812594116
