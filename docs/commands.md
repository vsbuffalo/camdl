# camdl commands: taxonomy and workflows

camdl's commands divide cleanly by what they _produce_. A model author moves
through them in a predictable order — author and check a model, simulate it
forward, diagnose whether the data can identify its parameters, fit it, then
read and compare the results. This document is the map: what each command makes,
which commands chain into which, and where the boundaries are.

The organizing distinction is **artifact-producing** vs **read/display**. The
first group runs a computation and stores a content-addressed run: `simulate`,
`pfilter` and `fit run` in the run store (`./camdl-store` unless `--store`, a
fit file's `store` key or `CAMDL_STORE` names another root, in that order; run
spec §2.1), and `profile` there too. The second group reads those runs back or
performs a pure utility transform. A few commands delegate to the compiler.

## The command map

### Produce simulation artifacts

| Command                  | Produces                                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `simulate` (alias `sim`) | One forward trajectory (+ optional synthetic observations, event log), or a grid of them — see [Sweeps and scenario grids](#sweeps-and-scenario-grids-simulate---sweep). |

### Write a design

| Command      | Writes                                                                                                                                                                                                                                                                                         |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `design lhs` | A Latin-hypercube design over a `fit.toml`'s `[estimate]` parameters, as a TSV that `simulate --sweep-file` reads. A function of the model, the fit config, `--n` and `--seed`; nothing is written to the store. See [Sweeps and scenario grids](#sweeps-and-scenario-grids-simulate---sweep). |

### Produce inference artifacts

| Command   | Produces                                                                                                                                                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit run` | Fit a `fit.toml`'s problem with its one `[method]` — an MLE (`if2`, the NLopt optimizers), a posterior (`pgas`, `pmmh`, `mh`) or a filter evaluation (`pfilter`), with its diagnostics. The production path.                                                  |
| `pfilter` | A log-likelihood at _fixed_ parameters via a bootstrap particle filter (no estimation); with `--sweep-file`, one per row of a design file — see [Evaluate the model over a parameter design](#evaluate-the-model-over-a-parameter-design).                    |
| `profile` | A profile-likelihood curve over a grid of one or more focal parameters — per-cell IF2 (`chain_binomial`) or an `nl-*` optimizer (`ode`).                                                                                                                      |
| `predict` | Bands of a stored fit run's posterior (`camdl predict @run`), or of one point given with a model file (`camdl predict MODEL --params FILE --data ...`): free-forward, one-step or a forecast — see [Predict from a fixed point](#predict-from-a-fixed-point). |
| `score`   | A stored prediction scored against the observed outcomes of the rows it predicted: log score, CRPS, WIS, PIT and interval coverage per row, and their means per stream, mode, role and lead — see [Score a prediction](#score-a-prediction).                  |

### Read, display, compare

| Command                                            | Does                                                                                                                                                                                                |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `list`                                             | Browse cached runs as a table; fits one row per fit run.                                                                                                                                            |
| `show`                                             | Full metadata for one cached run; `--format json` writes it as JSON.                                                                                                                                |
| `cat`                                              | Print a stored file of a result named by a handle or an output manifest (`camdl cat @m0 draws`); `--each` prints one TSV file of every result a manifest or label lists, each row after its labels. |
| `path`                                             | Print the path of a stored file of a result named by a handle or an output manifest; `--all` lists every file, `--each` one row per result with its run id and labels.                              |
| `compare`                                          | Paired comparison of stored scores (elpd, CRPS, WIS, PIT) across fits, one table per role and lead; computes and stores any score it needs. In-sample scores carry their optimism caveat.           |
| `diff`                                             | What differs between two model files, declaration by declaration, with whether the two compile to the same model. Exit 0 identical, 1 different, 2 no report.                                       |
| `label`                                            | Refuses every handle: a stored result is sealed. A result is named when it runs, with `--label`.                                                                                                    |
| `fit {run,summary,predict,diff,table,new,methods}` | Run a fit; inspect, summarize, predict-vs-observe, and aggregate fits; scaffold new `fit.toml`s.                                                                                                    |
| `dev eval`                                         | Evaluate model expressions (parameters, forcings) on a time grid — pure inspection, no simulation.                                                                                                  |
| `lineage {realize,tree,sojourn,cohort}`            | Offline projections over an event log — transmission tree, dwell times, cohort incidence.                                                                                                           |

**Listing fit runs.** `camdl list` prints fits in a `fits` table, one row per
fit run: when it was created, its run id, its problem's id (`FIT_ID`), the label
of an invocation that holds it, the model, method, seed, best log-likelihood and
its folder. `--format json` writes one object per run:
`{best_loglik, kind, label, method, model, path, problem_id, run_id, seed}`.

**Run handles and problem handles.** A stored fit is named by `@label` (the
invocation `fit run --label` named), a run-id or problem-id prefix, or its
folder. `camdl show @m0` describes that run (its path, kind, run id, problem,
method, seed, length and files), and `camdl cat @m0 draws` reads one of the
run's files, by its logical name (`draws`) or its stored name (`draws.tsv`);
without a name, `cat` on a run prints its `fit_state.toml`. A command that reads
one run refuses a problem handle and refuses a `fit.toml`, which never names a
run, listing the runs the file made. A `camdl predict` output is stored inside
its run, at `predict/<mode>-<12 hex>/`.

**Reading a stored file from a workflow.** A command run with
`--manifest build/fit.json` writes an output manifest naming every result it
stored by run id (run spec §2.10). A later step finds a file through it:
`camdl path --manifest build/fit.json draws` prints the file's absolute path,
after checking that the store holds the bytes the manifest records, and
`camdl cat --manifest build/fit.json draws` prints them. Over a sweep,
`camdl path --manifest build/sweep.json traj --each` prints one row per cell:
its run id, its labels and the path. The lookup rules are run spec §4.5.

**Browsing results as plain files.** `--link-root DIR`, else the environment
variable `CAMDL_LINK_ROOT`, makes `simulate`, `fit run`, `pfilter`, `profile`,
`predict`, `score` and `receive` also write a symbolic link to each stored
result file under DIR, at `DIR/<group>/<label>/<file>`: `fits/`, `sims/`,
`filters/`, `predictions/`, `profiles/` and `scores/`, then the command's
`--label`, then each file's logical name with its extension. The links under a
root follow the latest command with the label that succeeded with that root: a
command removes its label's links there, in every group, when it starts and
writes the new ones when it succeeds, so a failed command leaves none. A command
run without the root leaves the tree as it was. A workflow step that must read
exactly what an earlier step made reads through the manifest, whose run ids do
not move when the label is reused. The layout and the rules are run spec §2.11.

```bash
camdl fit run fit.toml --label m0 --manifest build/m0.json --link-root results
ls results/fits/m0/        # chain_0/  chain_1/  draws.tsv  fit_state.toml  …
```

A relative root is relative to the directory the command runs in. To keep one
root for a whole project, set the store and the link root in the project's
`.envrc`, which direnv reads after `direnv allow`; `$PWD` there is the directory
holding the file, so the paths hold from any subdirectory:

```bash
# .envrc
export CAMDL_STORE="$PWD/camdl-store"
export CAMDL_LINK_ROOT="$PWD/results"
```

**Selecting chains.** Every reader of a posterior cloud takes `--chains IDS`
(keep only those chains) or `--exclude-chains IDS` (drop those), never both:
`fit summary`, `camdl predict`, `fit table --quantity`,
`simulate --draws posterior` and `cat <run> draws`. `compare` takes either per
fit (`--chains @a:0,1`, `--exclude-chains @b:4`) or cohort-wide (bare ids); one
fit named in both flags, or a cohort-wide value beside a per-fit one, is
refused. Chain ids count from 0. A list that is empty, repeats an id or names an
id the fit lacks is refused, as is a selection that keeps every chain or none. A
selection is a view and never re-keys a run; on a six-chain fit
`--chains 0,1,2,4` and `--exclude-chains 3,5` are one selection, with one stored
prediction.

### Compiler passthrough

`check` and `inspect` (and `camdl dev compile`) delegate to `camdlc` — type- and
dimension-check a `.camdl` model, print its compiled structure, and compile it
to IR. `check` reports diagnostics (errors / warnings / lints);
`inspect --summary` prints the structural overview.

## Which methods go through `fit run`

`fit run` is the home for _fitting_. Every estimation and posterior-sampling
method is a `[method]` in a `fit.toml`; some also have a standalone command for
quick one-shot use.

| Method                   | `[method]` (`algorithm = …`) | Standalone command | Role                                                                       |
| ------------------------ | ---------------------------- | ------------------ | -------------------------------------------------------------------------- |
| IF2 (iterated filtering) | `"if2"`                      | —                  | Maximum-likelihood point estimate.                                         |
| PGAS + NUTS              | `"pgas"`                     | —                  | Bayesian posterior (exact complete-data likelihood, analytical gradients). |
| PMMH                     | `"pmmh"`                     | —                  | Bayesian posterior, gradient-free.                                         |
| Particle filter          | `"pfilter"`                  | `camdl pfilter`    | Log-likelihood evaluation at fixed θ (diagnostic).                         |
| NLopt (Subplex / BOBYQA) | `"nl-sbplx"`, `"nl-bobyqa"`  | —                  | Deterministic MLE on the ODE backend.                                      |
| Profile likelihood       | —                            | `camdl profile`    | Meta-routine running IF2 per grid point.                                   |

Two methods are **fit-run only** — PGAS and PMMH have no standalone command,
because Bayesian sampling presupposes the priors and convergence gates that a
`fit.toml` provides. Both **require priors** (declared under
`[estimate.<name>.prior]`); IF2, particle filter, and the NLopt optimizers do
not. PGAS uses gradient-based NUTS by default, which requires the compiler to
have emitted `rate_grad` expressions (autodiff); set `use_nuts = false` for a
gradient-free Metropolis-within-Gibbs fallback.

One routine is **standalone only**. `profile` is a meta-fit that maximizes the
likelihood at each point of a parameter grid (IF2, or an `nl-*` optimizer); it
orchestrates fits rather than being a single method, and prints one table row
per grid point (TSV, or `--format json`).

## Workflows

### Author → check → simulate

```bash
camdl check model.camdl                       # dimension + type check
camdl simulate model.camdl --param beta=0.4 --backend chain_binomial
camdl inspect model.camdl --summary           # compiled structure
camdl inspect model.camdl --cost-report       # per-step eval cost analysis
```

`simulate` is forward only: a model, parameters, a scenario, a seed → a
trajectory. Each cell is stored in the run store; `-o FILE` or `--stdout`
exports the combined table rendered from the stored cells, and
`camdl cat @LABEL` renders it again for a run named with `--label`. With `--obs`
it also draws synthetic observations through the model's observation block,
stored inside each cell.

### Evaluate the model over a parameter design

::: {.practice collapse="false" title="camdl survey has been removed"}

`camdl survey`, which scored a log-likelihood at every point of a
Latin-hypercube design, is no longer part of camdl. `camdl design lhs` writes
the design, and scoring a whole design is `camdl pfilter --sweep-file`. The
landscape summaries survey printed (the pair plot, the bound pile-up and
dimension warnings) may come back later, in camdl or in a separate tool.

:::

Before a long fit, run the model across the box of `[estimate]` bounds:

```bash
camdl design lhs model.camdl --fit fit.toml --n 200 --seed 1 -o design.tsv
camdl simulate model.camdl --sweep-file design.tsv --params fixed.toml
```

`design lhs` writes a Latin-hypercube design, a TSV with `point_id` and one
column per estimated parameter; `simulate --sweep-file` runs the model at every
row, with `fixed.toml` giving the parameters the design does not vary.

`pfilter --sweep-file` scores every row and writes one log-likelihood row per
design point:

```bash
camdl pfilter model.camdl --fit fit.toml --sweep-file design.tsv \
    --particles 1000 --replicates 5 --seed 1 -o landscape.tsv
```

Each point is scored exactly as the single `pfilter` call with the point's
values passed as `--param` would score it, on the same seeds, and is stored as
that call's result. So any row can be reproduced by one plain `pfilter` call, a
point already stored is reported `cached` rather than recomputed, and a design
run that was interrupted computes only the points it had not reached. The points
share random numbers, so their errors are correlated. `--fit` supplies the data,
the likelihood and the fit's `[fixed]` values.

The table has `point_id`, one `sweep:<param>` column per design column, `loglik`
(the mean over replicates), `loglik_sd` (empty under `--replicates 1` or when a
replicate failed), `n_failed`, `failure` and `run_id`. A replicate fails when it
scores `-inf` or the filter stops with an error; a point whose every replicate
failed is a row with `loglik = -inf` and its first failure message, stored like
any other point, and the run fails only when every point failed. A `--param` or
`--scenario` that would override a design column is refused, as is every per-run
output (`--trace`, `--save-paths` and the rest). Full reference: run spec §5.5,
"Scoring a design".

The points are not a posterior sample and are not a source of chain starts: a
fit whose chains should begin somewhere informed starts `from_prior` (the
default when every parameter has one), or from a short fit's posterior.

### The fit: one problem, one method

A `fit.toml` is a problem — model, data, what to estimate, what to fix, priors —
and one `[method]` that fits it. **One `pgas` method, started from the priors,
is the ordinary shape** — a Bayesian fit does not need an optimizer to find the
mode first, and starting from a point estimate concentrates the chains before
they have earned it.

A pipeline is two files run in turn: a cheap pass to rule a region out, a coarse
fit whose posterior seeds a finer one. The second file names the first by
handle, and says what kind of start it takes:

```toml
[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 1000
sweeps = 1000
starts = { from_posterior = "@coarse" } # one draw of @coarse's posterior per chain
```

```bash
camdl fit run fit.toml                              # the file's one [method]
camdl fit run fit.toml --starts from_prior          # override where the chains begin
camdl fit run fit.toml --resume <base-run-id>       # extend a completed run
camdl fit summary @label                            # Â or R̂ / verdict / MLE table
camdl predict @label --stream onset       # predicted-vs-observed artifact
```

`camdl predict` resolves the fit's posterior draws and writes
`predictive/<stream>.tsv` (the `q05…q95` ribbon, with typed `mode`/`treatment`
columns, a `role` and a `quantity` column, the fit's
`fit_rhat_max`/`fit_ess_min` stamp, and the per-row `rhat_mean`/`rhat_pred`
convergence channels) and `observed/<stream>.tsv` under the run directory. Join
the two on `(time, <dims>)` and plot, one facet per stratum. `role` is `trained`
for a row of the fit's training prefix (an `NA` row included), `held_out` for a
row of the bound file after that prefix (`[data] holdout_after`), and `future`
for a free-forward row past the last row of the bound file: this prediction read
no data row for this period (a `[data.holdout]` file is not read, so the periods
it covers are `future` too, gh#1309). A fixed-point prediction given no
`--holdout-after` labels its data rows `unknown`. `quantity` is `observation`:
the band is of draws from the observation model. For a stream labelled by each
period's start, `one_step` rows carry the period's end in `time` while
`free_forward` rows and `observed/` carry the label (gh#1152), and the
`free_forward` rows past the data are mis-timed (gh#1151); `role` is read by row
position and is right on both. The `one_step` band covers the training rows
only: a stored fit keeps its held-out rows' dates and not their values, and
filtering through a row needs its value. A forecast from the end of the training
rows is `--horizon`. Omit `--mode` for all applicable horizons (chain-binomial →
`free_forward` + `one_step`; ODE → `free_forward` only); an optimizer fit (IF2 /
NLopt) is refused since it has no posterior cloud.

Every mode predicts each stream the fit bound data to. A stream the model
declares but the fit bound no data to is _skipped_: it is named on stderr and in
`report.json` with the reason `no_bound_data`, and the exit status stays 0.
`--stream NAME` (repeatable) restricts the run to the named streams; a named
stream that cannot be produced — no bound data, or refused — makes the exit
status 1, and the streams that could be produced are still written. The manifest
(`predictive.json`, or `forecast.json` for a forecast) lists the streams
`produced`, `skipped` (each with its reason) and `failed`, so a reader can tell
what the artifact covers without the exit status. Every mode draws with the
binomial sampler the fit ran (`btpe` or `btrs`, read from the fit run's
`run_inputs.json`) and records it as `binomial` in the manifest.

`camdl predict @label --horizon "4 weeks"` (or `--mode forecast`) writes the
fixed-origin forecast instead: from the end of the fit's training rows (the
_forecast origin_), the state is filtered through the training rows only and
each stream's periods ending within the reach are predicted, the period under
way at the origin included. A target's `horizon` counts its stream's own periods
from that one (0, 1, …); `lead` is the time from the origin to the target's end.
The targets are the periods the stream declares — a uniform `covers` form
continues its tiling, window columns use `emit_schedule` — and the rows after
the training rows (the _held-out_ rows) only select which of those targets have
an observed value; they never renumber them. A held-out row within the reach
whose period is not one of the stream's targets is refused, not skipped, and so
are two held-out rows naming one target. A stream with window columns and no
`emit_schedule` takes its held-out rows' windows within the reach as its
targets, which must be in order, distinct and disjoint, and must not overlap its
training rows; their `horizon` counts the held-out rows, not declared periods,
so compare such targets by `lead`. A stream whose rows and `emit_schedule` give
different periods over the same reporting interval is refused, with both periods
and where each comes from. A stream with no declared rhythm and no held-out rows
is refused. A refused stream is listed under `failed`, named in `report.json`,
and the exit status is 1. A stream none of whose periods ends within the reach
is skipped with the reason `no_target_within_horizon` (a failure if `--stream`
named it).

A stream whose likelihood reads a design column, such as a binomial's
`n = tested`, needs that column's value at every target. `--design SOURCE=FILE`
(repeatable) supplies it from a file in the stream's own format holding its time
(or window) columns and the design columns, with no outcome column: a
_prospective_ forecast. `--realised-design` reads it from each target's held-out
row instead: the result is labelled _a conditional score using realised future
design_, since those values were known only after the origin, and is not an
unconditional forecast. A target with no value of a design column is refused,
never filled in; without either flag such a stream is refused. The target set,
with the design values the targets read, is part of the output address; the
held-out outcomes are not. The forecast's files record nothing about held-out
rows, so a forecast whose held-out rows change, or are removed, writes the same
files, byte for byte, to the same directory: which targets have a held-out row,
which row it is and its value belong to the score artifact. A held-out value is
scored only where it can be: a prospective forecast scores none of a
design-reading stream's (its values were realised under their own design), and a
value the data checks refuse, such as a count above its trials, is left
unscored, never refusing the prediction.

A forcing read from a data table (an `interpolated` forcing, a population or a
birth series) is an input at every time, and camdl cannot tell whether its
values after the origin were fixed in advance, observed later, or not known. A
forecast that reads such a table after its origin is refused, before any filter
runs, naming the forcing, the origin, the reach and the date its table ends,
until `--covariate FORCING=STATEMENT` says which (repeatable, one per forcing):

```bash
camdl predict @sir --horizon "8 weeks" --covariate pop=prospective  # fixed or assumed at the origin
camdl predict @sir --horizon "8 weeks" --covariate pop=realised     # observed after the origin
camdl predict @sir --horizon "8 weeks" --covariate pop=hold         # the value at the origin, carried forward
```

`prospective` and `realised` read the table as it is; `realised` labels the
forecast _conditional on covariates observed after the origin_ in
`forecast.json`, an after-the-fact evaluation in the sense `--realised-design`
is. `hold` holds the forcing at the table's interpolated value at the origin
wherever its argument `t − lag` is after the origin, and leaves every value
before the origin unchanged; for a `linear` table the held value depends on the
first row after the origin. `hold` is refused for a cubic-spline table
(`method = spline`), whose value before the origin depends on its rows after it.
A forcing with a positive `lag` reads its table at `t − lag`, so it needs a
statement only once the reach exceeds the lag, and one whose lag is a parameter
always needs one; a spline table needs one whenever it has a row after the
origin. An indexed forcing is stated per level by its underscore-joined name
(`--covariate pop_p0=hold`), as `--sweep` names an indexed parameter. Periodic
forcings, `piecewise` forcings and `bspline`s need none, and a statement for one
is refused; a forecast whose reach passes a `piecewise` breakpoint says so in a
note. The statements are part of the prediction's identity, so the three
forecasts above are three predictions. The rules are in the run spec, §2.2.2,
"Forcings read after the origin".

It writes `forecast/<stream>.tsv` (one row per target: `t_start`, `t_stop`,
`mode`, `horizon`, `lead`, `quantity`, the mean and the `q05…q95` band),
`forecast-samples.tsv` (joint samples, each one trajectory across every target),
and `forecast.json` (the origin, the reach, the rows that conditioned the
parameters and the state, each target's design values and the design mode with
its label, and the calendar). Forecasts are available for chain-binomial
posteriors (PMMH, PGAS); an ODE posterior, `--enable`, `--disable`,
`--quantities` and `--by-chain` are refused in this mode for now.

`--scenario NAME` and `--sweep` add _arms_ to a forecast: the fitted model is
the first arm, each scenario and each sweep point another, and every table gains
a leading `scenario` column (and a `sweep:<param>` column per swept parameter).
Every arm runs the same posterior draws with the same seeds, so all of them
filter the training rows identically and start from the same particles at the
origin; they differ only by what the scenario does after it. An arm must
therefore differ from the fitted model only at or after the origin: a scenario
that switches on or off an intervention firing before the origin (or at a time
that depends on a parameter or on the state), and a scenario or sweep that
changes a parameter the model reads outside its interventions (in the dynamics,
the initial state, a forcing or an observation model), are refused before
anything runs. After filtering, each arm's observation record before the origin
(log-likelihood increments, ESS, and each particle's log-likelihoods and
predictive draws) is compared with the fitted model's, bit for bit, and any
difference fails the forecast. The latent states are not compared, so a
pre-origin difference no observation reads is caught only by the check before
the run.

`--particles N` sets the particle count per posterior draw of the filters
`fit
predict` runs, the one-step band's and the forecast's. The default is the
fit's own filter's count for a PMMH fit and 500 otherwise (PGAS's conditional
SMC count is not a bootstrap-filter count). The manifest records the count and
where it came from (`flag`, `fit_pmmh` or `default`); the count is part of the
output address, so each count writes to its own directory. `forecast.json` also
records, per draw, the effective sample size of the filter's weights at its last
assimilation before the origin (before resampling), its time and ESS/N: a
measure of the Monte Carlo reliability of the state the forecast starts from,
not of its accuracy. `--particles` where no filter runs (`--mode free_forward`,
or the default mode on an ODE fit, which has no one-step band) is refused.

With more than one arm the forecast also writes `forecast-contrasts.tsv`: per
arm and target, the arm's predictive mean (`mean`), the fitted model's
(`mean_fitted`), their difference `delta`, and `delta_mcse`, the Monte Carlo
standard error of `delta`. That error is how far `delta` would move were the
forecast rerun with other random numbers and posterior draws; it is not the
predictive uncertainty of either arm, nor the uncertainty of the scenario's
effect. It is computed from each draw's difference of particle means, over the
chains' effective sample sizes, so neither the particles of a draw nor the draws
of a chain are taken as independent; when a chain carries fewer than 10
effective draws it is withheld and `mcse_note` says why (raise `--n-draws`). No
band of paired differences is written: its width would be set by how the arms'
random numbers are coupled, which the scenarios do not determine.

A resumed fit reads the base run read-only and writes a _new_ run keyed on the
extended length. It is a distinct deterministic artifact — not bit-identical to
an uninterrupted fit of the same length (both are valid posterior samples).

**Where the chains begin.** `starts` is one key for one concept. A _spread_ rule
gives each chain its own start — `from_prior`, `uniform_unconstrained`, `lhs`,
`uniform`, or `{ from_posterior = "<handle>" }` — so the between-chain R̂ can say
whether the chains found the same posterior. A _point_ rule puts every chain at
one point — `single`, `{ from_mle = "<handle>" }`, `{ from_params = "<toml>" }`
— and R̂ is then reported as not assessed rather than as a pass. A handle is
`@label`, a run-id prefix, or the run's folder; a `fit.toml` names no run and is
refused.

A source that did not converge is refused where it is consumed: starting from an
unconverged fit launders its multi-modality into this one, and
`--allow-nonconverged-source` records the choice to do it anyway. Every
multi-chain sampler writes `chain_starts.tsv`, the record of where each chain
actually began.

### Predict from a fixed point

`camdl predict` takes a model file in place of a run handle to predict at one
point given on the command line: an estimate made elsewhere, an optimizer fit's
estimate (`camdl fit summary <run> --params-only`), a sampler fit's posterior
mean or median (`camdl fit summary <run> --point posterior-mean`), or values
from the literature.

```bash
camdl fit summary @post --point posterior-mean > mean.toml
camdl predict seir.camdl --params mean.toml --data cases=cases.tsv
```

`--point` takes each estimated parameter's mean (or median) over the run's
posterior draws, every chain pooled, on the parameter's natural scale; a mean
taken on the log or logit scale is a different number. The file's `[provenance]`
block records the run, the statistic, the number of draws, and the backend and
`dt` the fit ran on, which `camdl simulate --params mean.toml` replays on. The
point is one no chain visited: a band predicted at it carries process and
observation noise only, labelled `plug_in`, and leaves out the parameter
uncertainty the posterior predictive of `camdl predict @post` carries.

```bash
camdl predict seir.camdl --params mle.toml --data cases=cases.tsv \
  --holdout-after 2026-03-01 --horizon '4 weeks' --scenario school_closure
```

The point is the model's values, each `--params` file in order, then each
`--param NAME=VALUE`; a parameter with no value after all of them is refused.
`--data` is required in every mode, bound as `pfilter` binds it, and the streams
and rows predicted are the bound streams' rows. `--holdout-after` has the
grammar and the cut of `fit.toml`'s `[data] holdout_after`: the rows after it
are parsed to find the cut, so a malformed one is refused, and are then neither
stored nor used; a forecast starts at the end of the rows before it. A
`last_obs - …` cut resolves over the whole bound files, held-out rows included,
as a fit's does. Without it every bound row is conditioned on. `--table`,
`--backend`, `--dt`, `--integrator`, `--binomial` and `--allow-degenerate-rates`
mean what they mean for `simulate` and `pfilter`. `--time-format` checks each
`--data` file's time column against the format given and refuses one it does not
hold; the reads after it detect the format from the column, so it does not
silence W326.

The point is repeated `--n-draws` times (default 200), each copy under its own
seed derived from `--seed`, and every mode pools over the copies as it pools
over a posterior's draws: free-forward runs one replay per copy, one-step and
forecast one filter per copy, pooled over particles (default 500) and copies.
The bands therefore carry process and observation noise and no parameter
uncertainty. Every row's `treatment` is `plug_in`, the arm with no scenario is
named `(none)`, and the columns that compare chains or count effective draws are
empty. `camdl show` says that the bands carry no parameter uncertainty.

The prediction is stored at `prediction/<model>-<12 hex>/<mode>-<12 hex>/`,
grouped by the compiled model, with the compile and each stream's conditioned
rows stored as the results it depends on. A repeat with the same inputs reuses
it without computing it again.

Both forms take the store flags every stored command takes. `--store DIR` names
the run store, over `CAMDL_STORE` and `./camdl-store`: a fixed-point prediction
is stored there, and a run handle is looked up there (a run named by its folder
is predicted in the store holding it). `--label NAME` names the invocation, so
`camdl show @NAME` finds the prediction. `--force` computes a stored prediction
again, moving the stored one to the store's `.quarantine/`. `-o DIR` exports a
copy of the prediction's result files into `DIR`, at their paths in its folder,
whether the prediction was computed or reused.

The flags that read a posterior (`--by-chain`, `--exclude-chains`, `--chains`,
`--realised-design`) are refused with a model file, and the flags that pose a
fixed point are refused with a run handle, whose problem holds its model, data
and numerics. The other refusals are those of a posterior prediction: one-step
on ODE or Gillespie, a forecast on Gillespie, `--particles` where no filter
runs, `--design` and `--covariate` outside a forecast.

### Score a prediction

`camdl score` scores a stored prediction against the outcomes of the rows it
predicted. It takes the prediction's handle and one outcome file per stream
source, and nothing else: the prediction's parameter source, mode, reach,
particles and seed are in its identity.

```bash
camdl predict @sir --mode forecast --horizon "8 weeks" --label sir_fc
camdl score @sir_fc --data cases=london.tsv
```

For a stored fit the outcome file is the file the fit was given, held-out rows
included; for a fixed point it is the file the prediction was given, extended
past `--holdout-after`. Every row the prediction conditioned on must be in it
unchanged, or the score is refused naming the first row that differs. The score
computes the prediction again with the outcomes bound and checks that it writes
the stored prediction's files byte for byte before storing anything, so a score
costs about what its prediction cost.

The score is stored inside the prediction, at `scores/score-<12 hex>/`:
`scored.tsv` (one row per scored row or target), `summary.tsv` (one row per
stream, mode, role and lead; a `joint` stream holds the log score of every
stream observed at one time) and `score.json`. A one-step score continues the
filter through the held-out rows, predicting each from the rows before it. The
files and the rules are in the run specification, §2.2.4.

### Sweeps and scenario grids: `simulate --sweep`

A grid is forward simulation only — **no inference**. `simulate` runs the
Cartesian product of its parameter points × scenarios × replicates (or explicit
seeds) into the run store, one stored cell per grid point. No combined table is
stored: `-o FILE` and `camdl cat @LABEL` render it from the cells. The parameter
points are a _design_ you choose (`--sweep PARAM=GRID`, repeatable, or
`--sweep-file FILE`) or a _sample_ (`--draws`); the two are refused together.

```bash
# five beta values × two scenarios × fifty seeds, with common random numbers
camdl simulate sirv.camdl --params params.toml \
    --scenario baseline,with_vaccination --sweep beta=lin(0.2,0.6,5) --seeds 1:50

# two swept parameters form the name-sorted Cartesian product
camdl simulate sir.camdl --params p.toml --sweep beta=0.2,0.3 --sweep gamma=0.1,0.2 --replicates 10

# design points from a TSV (header = parameters, one row per point)
camdl simulate sir.camdl --params p.toml --sweep-file design.tsv --replicates 5

# a 200-point Latin-hypercube design over fit.toml's [estimate] parameters, then run it
camdl design lhs sir.camdl --fit fit.toml --n 200 --seed 1 -o design.tsv
camdl simulate sir.camdl --params p.toml --sweep-file design.tsv

# plan against the store without running: every cell a hit or a miss
camdl simulate sirv.camdl --params params.toml --sweep beta=lin(0.2,0.6,5) --seeds 1:50 --dry-run

# the observations of every cell as a dataset: one TSV per stream under synth/
camdl simulate sirv.camdl --params params.toml --sweep beta=lin(0.2,0.6,5) --seeds 1:50 --obs-dir synth
```

Every design point at a given replicate shares one random stream, so paired
differences across points are low-variance and adding a point leaves the others
unchanged (for independent noise per point, run one invocation per point with
its own `--seed`). The combined table and the observation tables carry
`replicate`, `scenario`, `point_id`, one `sweep:<param>` column per swept
parameter, and `seed` — the pairing key across points, and across scenarios. Do
not pair scenarios on `replicate`: it numbers the cells of the whole run, so it
runs on from one scenario to the next and a join on it matches nothing; with
`--draws`, pair on `draw` (run spec §10.2). A scenario that sets or scales a
swept parameter, or a `--param` on one, is refused. A second run of the same
grid reads every cell the store holds back instead of re-simulating it, under
any `--label`; `--force` recomputes. A cell whose engine stops with an error is
stored with its `failure.toml` and reused by a rerun; the run lists every such
cell and exports nothing. The arm of an ad-hoc run is labelled from its flags
(`(none)` for no flags, `(none)+school_closure` for `--enable school_closure`,
`(none);beta=0.3` for `--param beta=0.3`).

Use a sweep to explore _forward_ behavior under interventions and parameter
ranges. To estimate parameters from data, use `fit`.

### Model comparison

```bash
camdl compare @sir @seir --data cases=london.tsv     # held-out rows, one-step
camdl compare @sir @seir --in-sample                 # training rows
camdl compare @sir_score @seir_score                 # two stored scores
camdl compare a/prequential.json b/prequential.json --baseline a
camdl compare @baseline @candidate --pointwise pointwise.tsv
```

`compare` renders a baseline-centered table of stored scores: per row its elpd
(the sum of the joint per-step log scores), and per stream its CRPS, WIS and
coverage — in the table for a single-stream comparison, in a per-stream block
below it otherwise; no figure combines streams. Each argument is a stored score
(the `@label` of a `camdl score`, a run-id prefix or its folder), a stored
prediction, which is scored, or a stored fit run, which is predicted and scored;
or a `prequential.json` (or a directory holding one), written by a `pfilter`
method or by `camdl pfilter --save-prequential` and read **as-is**. A prediction
`compare` makes from a fit run is `camdl predict RUN --mode MODE` with the
`--particles` (default 1000) and `--seed` given to `compare`, the same for every
fit run compared; it and its score are computed in the process and stored, and
found again by the next `compare` or by `camdl predict` with the same options.
`--draws` reaches only a sampler's run, which is predicted from that many
posterior draws, of which `--draws 1` is one draw; an optimizer run (IF2, NLopt)
is predicted once, at its winning point, labelled plug-in, whatever `--draws`
is. A stored prediction or score read in a filtered mode must have run its
filters at compare's particle count when `--particles` is given or a fit run is
predicted, and stored inputs alone at one count; otherwise they are refused
before anything is computed. Each input is predicted `--replicates` times
(default 5) at successive seeds (a stored prediction's from its own seed, with
its other options), each replicate scored against the outcome rows of the first,
which gives each row a filter-noise SE; with `--replicates 1` the evidence
column says the filter noise was not measured. An ODE free-forward score does
not depend on the seed, and is predicted once, with no SE. `--mode` picks the
mode compared, one-step by default where the backend runs a particle filter and
free-forward otherwise; `--mode forecast` needs `--horizon`, and compares each
lead in a table of its own.

A score needs the outcomes of the rows it scores. A stored fit run keeps its
held-out rows' dates and not their values, so its held-out rows are compared
from `--data STREAM=FILE` (the files the fit was given, each read by every fit
that binds the stream), from `--data-held-out STREAM=FILE` (a file of the
held-out rows alone, such as a `[data.holdout]` file), or from a score of the
prediction already stored; without any, `compare` refuses and says which. The
two flags give one score for one outcome, and one stream takes one of them.
`camdl score` takes both flags too. `--in-sample` compares the training rows
(role `trained`), which need no file, as does a fit that held nothing out. Every
row shows its `role`, and a forecast's its `lead`.

An explicit trace below schema 7, written by an older build before held forcings
were recorded, is refused: it cannot say whether a forcing was held outside its
range while it was scored. Re-run the fit or the `pfilter --save-prequential`
that wrote it. In-sample scores are **optimistic**: the parameters were fitted
to the scored rows, so they are not a forecast score. The optimism does not
cancel when two models are differenced, either: it grows with the effective
number of parameters each model was free to tune against the same observations,
so Δelpd tilts toward the more flexible model. Read a comparison as indicative,
alongside `se(Δ)` and the caveats.

The evidence column is gated on that standard error: when `|Δelpd| < 2·se(Δ)`
the cell reads `within noise` and carries no Jeffreys tier, because the tier
would name a magnitude the data cannot resolve. The `LR` column is `exp(Δelpd)`
— the candidate's predictive likelihood over the baseline's on the same scored
observations.

The footer under the table states what is specific to that comparison: the
scored window and baseline, the conditioning the rows were scored under
(in-sample, or a held-out tail with the sealed `train_end`), the optimism
caveat, any miscalibration flag, and the numbers the noise gates read — the
filter-noise MC SE per row (with `--replicates`) and the lag-1 autocorrelation
of the per-step Δelpd. What those quantities are, and their citations and
caveats, is `camdl compare --explain` (the same guide as
`camdl docs model-comparison`).

`compare` refuses, rather than renders, these unlike-for-unlike comparisons,
naming the field that differs, with no override: scores of rows predicted in
different modes, of different roles (training rows against held-out ones), at
different forecast leads, of forecasts whose reach is written differently, or
whose covariates after the origin are stated differently (`realised` against
not); a stored fit's verified held-out rows against a fixed point's declared
ones; and rows on different predictive bases (a filter's log-likelihood estimate
is biased downward and an ODE score is exact). It also refuses a differing
`T_score` (override: `--allow-mismatched-horizon`, which suppresses the Δ
columns and waives only the times one row scored and the other did not — at
every time both scored, the stream and window checks below still apply), traces
scored at different observation times, traces that scored different observation
streams at the same time (gh#570 — the step's log score is the joint score over
those streams), traces that scored the same stream at the same time over
different _windows_ (gh#833 — a gapped reading of a file against a merged one,
or a stream read as an instant against the same name read as a flow, produce
different likelihoods at identical times), rows conditioned on different
observed data or scoring different observed values (override:
`--allow-data-mismatch`, which renders the Δ as confounded), and traces scored
with different particle or replicate counts (override:
`--allow-mismatched-filter-settings`). Provenance kinds that differ (plug-in
against a posterior mixture) are refused too (override:
`--allow-mixed-provenance`). Under each override that was needed, every format
states it once — the flag, the rows it concerned and what it does to the reading
— as a line under the table and an entry of the JSON's `overrides` array.

`--pointwise PATH` writes the per-observation difference the table already
computes in order to form `se(Δelpd)`, as two TSV files. `PATH` holds one row
per candidate × scored step with the joint score — the log of the joint
predictive density of every stream observed at that time, the quantity elpd
sums: `model`, `baseline`, `t`, `log_score`, `baseline_log_score`,
`delta_log_score`. A sibling file, named by inserting `.streams` before the
extension (`scores.tsv` → `scores.streams.tsv`; a path with no extension gets
`.streams` appended), holds one row per candidate × step × stream with that
stream's own score: `model`, `baseline`, `t`, `stream`, `log_score`,
`baseline_log_score`, `delta_log_score`. The two are separate because a stream's
score is not a share of the joint one: the per-stream rows at a time do not sum
to the joint row. An empty cell in the per-stream file means that side did not
score that stream at that time. `Δelpd = 12 nats` says a model won; these files
say _where_ it won — on three weeks around an intervention, on one district, on
a single reporting batch. An elpd gap taken across two different stream sets
never reaches them: the preflight refuses that comparison before the table is
rendered or either file is written, naming the streams that differ (gh#570).

```bash
camdl compare @full @cases_only --streams cases
```

`--streams NAME[,NAME...]` (also `streams = [...]` in `compare.toml`) compares
the fits on chosen observation streams: one table per stream, each scoring every
row on that stream's own one-step-ahead predictive, read from the per-stream
scores the trace already stores. A name is a stream of the model or a data
source, which selects every stream it expands to. This is how fits that bound
different streams — the model fitted with and without a stream — are compared on
a stream they share; the stream-set refusal above names the option and the
shared streams. Each table states its stream and, when the rows' fits
assimilated different streams, each row's set; the data check compares only that
stream's source. No figure combines streams: two selected streams give two
tables and nothing else, the JSON is
`{ "streams_scored": [...], "tables": [...] }`, and `--pointwise` writes only
the `.streams` file. Fits that assimilated different streams and are scored
in-sample are refused, because that comparison leans toward the fit trained on
fewer streams; declare the same `holdout_after` in every fit, or pass
`--allow-in-sample-ablation` to render with that caveat stated. The methods are
in §7 of `camdl docs model-comparison`.

## The boundaries, stated plainly

- **A forcing is not extended past its range silently.** A run that evaluates an
  `interpolated` forcing before its first or after its last data point, or a
  `bspline` outside its window, is refused by every command that runs a model
  (`simulate`, `fit run`, `camdl predict`, `pfilter`, `profile`, `compare` when
  it predicts, `dev eval`). The error names the forcing, its range and the side
  it was evaluated on. `--allow-forcing-extrapolation` on any of them (in a fit
  config, `[config] allow_forcing_extrapolation = true`) holds the forcing's end
  value instead, prints one warning per forcing naming its range and the span
  held, and records the override in each `run.json` the command writes. The
  override is part of the run's identity, so a run under it and the same run
  without it are two runs: neither is served the other's stored result.

- **`simulate` goes forward; `fit` goes backward.** Simulation maps parameters →
  data; fitting maps data → parameters. A sweep is many forward runs, never an
  inference.
- **`profile` is about the likelihood _surface_, not a fit.** It traces the
  maximized likelihood along one axis. It can precede or contextualize a fit; it
  does not produce a posterior.
- **`pfilter` is a shortcut into machinery `fit run` also uses.** The same
  particle-filter core backs both surfaces; the standalone command exposes a
  smaller set of knobs for a quick fixed-θ log-likelihood, while a `fit.toml`
  `[method]` adds a start rule, the convergence diagnostics, and post-fit
  audits. An MLE-only run is not a separate command — it is a `fit.toml` whose
  `[method]` is `algorithm = "if2"`, run through `camdl fit run`.
