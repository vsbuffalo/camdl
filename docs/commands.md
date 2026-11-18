# camdl commands: taxonomy and workflows

camdl's commands divide cleanly by what they _produce_. A model author moves
through them in a predictable order — author and check a model, simulate it
forward, diagnose whether the data can identify its parameters, fit it, then
read and compare the results. This document is the map: what each command makes,
which commands chain into which, and where the boundaries are.

The organizing distinction is **artifact-producing** vs **read/display**. The
first group runs a computation and writes a content-addressed run under
`results/`; the second group reads those runs back or performs a pure utility
transform. A few commands delegate to the compiler.

## The command map

### Produce simulation artifacts

| Command                  | Produces                                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `simulate` (alias `sim`) | One forward trajectory (+ optional synthetic observations, event log), or a grid of them — see [Sweeps and scenario grids](#sweeps-and-scenario-grids-simulate---sweep). |

### Produce inference artifacts

| Command   | Produces                                                                                                                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fit run` | Fit a `fit.toml`'s problem with its one `[method]` — an MLE (`if2`, the NLopt optimizers), a posterior (`pgas`, `pmmh`, `mh`) or a filter evaluation (`pfilter`), with its diagnostics. The production path. |
| `pfilter` | A log-likelihood at _fixed_ parameters via a bootstrap particle filter (no estimation).                                                                                                                      |
| `profile` | A profile-likelihood curve over a grid of one or more focal parameters — per-cell IF2 (`chain_binomial`) or an `nl-*` optimizer (`ode`).                                                                     |

### Produce diagnostic artifacts

| Command  | Produces                                                                                                                                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `survey` | A likelihood landscape: log-likelihood at many Latin-hypercube points across a parameter box. Answers "is this model identifiable from this data?" _before_ a fit. Not a fitting routine. |

### Read, display, compare

| Command                                            | Does                                                                                                                                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `list`                                             | Browse cached runs as a table; fits one row per run, grouped by problem folder (`--history`, `--runs`).                                                                               |
| `show`                                             | Full metadata for one cached run; `--format json` writes it as JSON (`fit_show` for a fit run or a problem folder).                                                                   |
| `cat`                                              | Emit a cached run's trajectory or observations as TSV; `--stream draws.tsv --chains IDS` (or `--exclude-chains IDS`) prints only the selected chains' rows.                           |
| `compare`                                          | Paired prequential comparison (elpd, CRPS, PIT) across fits. Scores are plug-in + in-sample-optimistic (the caveat is printed with the table); not a leave-future-out forecast score. |
| `diff`                                             | What differs between two model files, declaration by declaration, with whether the two compile to the same model. Exit 0 identical, 1 different, 2 no report.                         |
| `label`                                            | Set a label on any run; a fit run labelled `L` becomes `@L`, and the run that held `L` before is `@L~1`.                                                                              |
| `fit {run,summary,predict,diff,table,new,methods}` | Run a fit; inspect, summarize, predict-vs-observe, and aggregate fits; scaffold new `fit.toml`s.                                                                                      |
| `dev eval`                                         | Evaluate model expressions (parameters, forcings) on a time grid — pure inspection, no simulation.                                                                                    |
| `lineage {realize,tree,sojourn,cohort}`            | Offline projections over an event log — transmission tree, dwell times, cohort incidence.                                                                                             |

**Listing fit runs.** `camdl list` prints fits one row per run, grouped under a
`problem` line per problem folder (its fit id, model, config and path). Each row
gives the run's label, run id, method, seed, how its chains started (`STARTS`:
the recorded source run id, with the handle as typed beside it, since the handle
may since have moved), its live status (`completed`, `running`, `interrupted`,
`failed`) and its stored verdict, the line `fit summary` leads with (`—` when
there is nothing to judge; `unreadable: <reason>` when the stored summary cannot
be read, which never fails the listing). Unlabelled runs of one method that
differ only in seed share a row with the seed range and a combined verdict;
`--runs` lists each. `--history` adds each label's earlier holders, as `m0~1`,
`m0~2`. `--limit` counts runs. `--format json` writes one `fit_list_row`
(schema 3) object per run:
`{schema, kind, problem {fit_id, model, config}, label, label_set_at, history_index, run_id, method, seed, starts {rule, handle, run_id, chains}, status, verdict, path}`.

**Run handles and folder handles.** `show` and `cat` resolve a fit handle the
way every fit verb does. `@label`, a leaf directory and a `run_id` prefix name
one run: `camdl show @m0` describes that run (its id, problem, method, how its
chains started, its stored verdict, path and recorded outputs, including its
predictions), and `camdl cat @m0 --stream draws.tsv` reads a file from the run's
leaf, then from its default-options predictions under the problem folder's
`predict/<run id>-<key>/`, then from the problem folder. A fit-id prefix, a
problem folder's path and a `fit.toml` name the folder: `show` lists its runs
and the handle that selects each, and `cat` on a folder holding several runs is
refused with them listed, unless the stream is a file of the folder itself
(`fit.meta.json`, or a `predict/<run id>-<key>/…` path). Without `--stream`,
`cat` on a run prints its `fit_state.toml` and on a folder its `fit.meta.json`.

**Selecting chains.** Every reader of a posterior cloud takes `--chains IDS`
(keep only those chains) or `--exclude-chains IDS` (drop those), never both:
`fit summary`, `fit predict`, `fit table --quantity`,
`simulate --draws posterior` and `cat --stream draws.tsv`. `compare` takes
either per fit (`--chains @a:0,1`, `--exclude-chains @b:4`) or cohort-wide (bare
ids); one fit named in both flags, or a cohort-wide value beside a per-fit one,
is refused. Chain ids count from 0. A list that is empty, repeats an id or names
an id the fit lacks is refused, as is a selection that keeps every chain or
none. A selection is a view and never re-keys a run; on a six-chain fit
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
| Survey (LHS landscape)   | —                            | `camdl survey`     | Identifiability diagnostic; _feeds_ a fit, is not a method.                |
| Profile likelihood       | —                            | `camdl profile`    | Meta-routine running IF2 per grid point.                                   |

Two methods are **fit-run only** — PGAS and PMMH have no standalone command,
because Bayesian sampling presupposes the priors and convergence gates that a
`fit.toml` provides. Both **require priors** (declared under
`[estimate.<name>.prior]`); IF2, particle filter, and the NLopt optimizers do
not. PGAS uses gradient-based NUTS by default, which requires the compiler to
have emitted `rate_grad` expressions (autodiff); set `use_nuts = false` for a
gradient-free Metropolis-within-Gibbs fallback.

Two routines are **standalone only**. `survey` is a diagnostic, not a fit — it
maps the landscape so you can see ridges and multimodality before committing
compute. `profile` is a meta-fit that maximizes the likelihood at each point of
a parameter grid (IF2, or an `nl-*` optimizer); it orchestrates fits rather than
being a single method, and prints one table row per grid point (TSV, or
`--format json`).

## Workflows

### Author → check → simulate

```bash
camdl check model.camdl                       # dimension + type check
camdl simulate model.camdl --param beta=0.4 --backend chain_binomial
camdl inspect model.camdl --summary           # compiled structure
camdl inspect model.camdl --cost-report       # per-step eval cost analysis
```

`simulate` is forward only: a model, parameters, a scenario, a seed → a
trajectory. With `--obs` it also draws synthetic observations through the
model's observation block.

### Diagnose identifiability: survey → fit

Before burning hours on a fit, map the likelihood:

```bash
camdl survey model.camdl --fit fit.toml       # LHS landscape → landscape.tsv
```

The survey is a diagnostic, not a starting-point source: a landscape is not a
posterior. A fit whose chains should begin somewhere informed starts
`from_prior` (the default when every parameter has one), or from a short fit's
posterior.

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
camdl fit summary results/fits/<dir>/               # Â or R̂ / verdict / MLE table
camdl fit predict --fit fit.toml --stream onset     # predicted-vs-observed artifact
```

`fit predict` resolves the fit's posterior draws and writes
`predictive/<stream>.tsv` (the `q05…q95` ribbon, with typed
`horizon`/`treatment` columns, the fit's `fit_rhat_max`/`fit_ess_min` stamp, and
the per-row `rhat_mean`/`rhat_pred` convergence channels) and
`observed/<stream>.tsv` under the run directory. Join the two on
`(time, <dims>)` and plot, one facet per stratum. Omit `--mode` for all
applicable horizons (chain-binomial → `free_forward` + `one_step`; ODE →
`free_forward` only); an optimizer fit (IF2 / NLopt) is refused since it has no
posterior cloud.

Every mode predicts each stream the fit bound data to. A stream the model
declares but the fit bound no data to is _skipped_: it is named on stderr and in
`report.json` with the reason `no_bound_data`, and the exit status stays 0.
`--stream NAME` (repeatable) restricts the run to the named streams; a named
stream that cannot be produced — no bound data, or refused — makes the exit
status 1, and the streams that could be produced are still written. The manifest
(`predictive.json`, or `forecast.json` for a forecast) lists the streams
`produced`, `skipped` (each with its reason) and `failed`, so a reader can tell
what the artifact covers without the exit status. Every mode draws with the
binomial sampler the fit ran (`btpe` or `btrs`, read from the fit leaf) and
records it as `binomial` in the manifest.

`fit predict --fit fit.toml --horizon "4 weeks"` (or `--mode forecast`) writes
the fixed-origin forecast instead: from the end of the fit's training rows (the
_forecast origin_), the state is filtered through the training rows only and
each stream's periods ending within the reach are predicted, the period under
way at the origin included. A target's `horizon` counts its stream's own periods
from that one (0, 1, …); `lead` is the time from the origin to the target's end.
The targets are the periods the stream declares — a uniform `covers` form
continues its tiling, window columns use `emit_schedule` — and a stream with
neither, or whose likelihood reads a data column such as a binomial's `n`, is
refused a forecast: it is listed under `failed`, named in `report.json`, and the
exit status is 1. A stream none of whose periods ends within the reach is
skipped with the reason `no_target_within_horizon` (a failure if `--stream`
named it). It writes `forecast/<stream>.tsv` (one row per target: `t_start`,
`t_stop`, `horizon`, `lead`, the held-out row it corresponds to, the mean and
the `q05…q95` band), `forecast-samples.tsv` (joint samples, each one trajectory
across every target), and `forecast.json` (the origin, the reach, the rows that
conditioned the parameters and the state, and the calendar). Forecasts are
available for chain-binomial posteriors (PMMH, PGAS); an ODE posterior,
`--enable`, `--disable`, `--quantities` and `--by-chain` are refused in this
mode for now.

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
anything runs. After filtering, each arm's record before the origin is compared
with the fitted model's, bit for bit, and any difference fails the forecast.

`--particles N` sets the particle count per posterior draw of the filters
`fit
predict` runs, the one-step band's and the forecast's. The default is the
fit's own filter's count for a PMMH fit and 500 otherwise (PGAS's conditional
SMC count is not a bootstrap-filter count). The manifest records the count and
where it came from (`flag`, `fit_pmmh` or `default`); a count other than the
default writes to its own output directory. `forecast.json` also records, per
draw, the effective sample size of the filter's weights at its last assimilation
before the origin (before resampling), its time and ESS/N: a measure of the
Monte Carlo reliability of the state the forecast starts from, not of its
accuracy. `--particles` with `--mode free_forward`, which runs no filter, is
refused.

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
`@label`, a fit-id or leaf `run_id` prefix, a leaf directory, or a `fit.toml`.

A source that did not converge is refused where it is consumed: starting from an
unconverged fit launders its multi-modality into this one, and
`--allow-nonconverged-source` records the choice to do it anyway. Every
multi-chain sampler writes `chain_starts.tsv`, the record of where each chain
actually began.

### Sweeps and scenario grids: `simulate --sweep`

A grid is forward simulation only — **no inference**. `simulate` runs the
Cartesian product of its parameter points × scenarios × replicates (or explicit
seeds) into the store, one leaf per cell plus one combined `ensemble.tsv`. The
parameter points are a _design_ you choose (`--sweep PARAM=GRID`, repeatable, or
`--sweep-file FILE`) or a _sample_ (`--draws`); the two are refused together.

```bash
# five beta values × two scenarios × fifty seeds, with common random numbers
camdl simulate sirv.camdl --params params.toml \
    --scenario baseline,with_vaccination --sweep beta=lin(0.2,0.6,5) --seeds 1:50

# two swept parameters form the name-sorted Cartesian product
camdl simulate sir.camdl --params p.toml --sweep beta=0.2,0.3 --sweep gamma=0.1,0.2 --replicates 10

# design points from a TSV (header = parameters, one row per point)
camdl simulate sir.camdl --params p.toml --sweep-file design.tsv --replicates 5

# plan against the store without running: every cell a hit or a miss
camdl simulate sirv.camdl --params params.toml --sweep beta=lin(0.2,0.6,5) --seeds 1:50 --dry-run

# the observations of every cell as a dataset: one TSV per stream under synth/
camdl simulate sirv.camdl --params params.toml --sweep beta=lin(0.2,0.6,5) --seeds 1:50 --obs-dir synth
```

Every design point at a given replicate shares one random stream, so paired
differences across points are low-variance and adding a point leaves the others
unchanged (for independent noise per point, run one invocation per point with
its own `--seed`). The combined table and the observation tables carry
`point_id`, one `sweep:<param>` column per swept parameter, and `seed` — the
pairing key across points, and across scenarios. Do not pair scenarios on
`replicate`: it numbers the cells of the whole run, so it runs on from one
scenario to the next and a join on it matches nothing; with `--draws`, pair on
`draw` (run spec §10.2). A scenario that sets or scales a swept parameter, or a
`--param` on one, is refused. A second run of the same grid reads every cell the
store holds back instead of re-simulating it; `--force` recomputes. The arm of
an ad-hoc run is labelled from its flags (`(none)` for no flags,
`(none)+school_closure` for `--enable school_closure`, `(none);beta=0.3` for
`--param beta=0.3`).

Use a sweep to explore _forward_ behavior under interventions and parameter
ranges. To estimate parameters from data, use `fit`.

### Model comparison

```bash
camdl compare results/fits/a/posterior results/fits/b/posterior --baseline a
camdl compare @baseline @candidate
camdl compare @baseline @candidate --pointwise pointwise.tsv
```

`compare` renders a baseline-centered table of prequential scores (the joint
per-step log score, and each stream's CRPS and PIT coverage — in the table for a
single-stream comparison, in a per-stream block below it otherwise; no figure
combines streams). Each argument is either an explicit prequential artifact — a
`prequential.json` (or a method directory holding one), written by a `pfilter`
method or by `camdl pfilter --save-prequential`, and read **as-is** — **or** a
fit handle (`@label`, a hash prefix, a run directory, or a `fit.toml`), whose
prequential is **auto-derived** by re-filtering at the fit's sealed θ̂.
`--particles` and `--seed` set the filter used for any auto-derived handle and
are applied **uniformly** across all derived fits, so `T_score` and the scores
stay commensurable; they are ignored for an explicit `prequential.json` (read
as-is). The scores are **plug-in and in-sample-optimistic** — computed at a
single θ fit to the whole series — so they are not a leave-future-out forecast
score. The optimism does not cancel when two models are differenced, either: it
grows with the effective number of parameters each model was free to tune
against the same observations, so Δelpd tilts toward the more flexible model.
Read a comparison as indicative, alongside `se(Δ)` and the caveats.

The evidence column is gated on that standard error: when `|Δelpd| < 2·se(Δ)`
the cell reads `within noise` and carries no Jeffreys tier, because the tier
would name a magnitude the data cannot resolve. The `LR` column is `exp(Δelpd)`
— the candidate's in-sample predictive likelihood over the baseline's on the
same scored observations.

The footer under the table states what is specific to that comparison: the
scored window and baseline, the conditioning the rows were scored under
(in-sample, or a held-out tail with the sealed `train_end`), the optimism
caveat, any miscalibration flag, and the numbers the noise gates read — the
filter-noise MC SE per row and the lag-1 autocorrelation of the per-step Δelpd.
What those quantities are, and their citations and caveats, is
`camdl compare --explain` (the same guide as `camdl docs model-comparison`).

`compare` refuses, rather than renders, these unlike-for-unlike comparisons: a
differing `T_score` (override: `--allow-mismatched-horizon`, which suppresses
the Δ columns and waives only the times one row scored and the other did not —
at every time both scored, the stream and window checks below still apply),
traces scored at different observation times, traces that scored different
observation streams at the same time (gh#570 — the step's log score is the joint
score over those streams), traces that scored the same stream at the same time
over different _windows_ (gh#833 — a gapped reading of a file against a merged
one, or a stream read as an instant against the same name read as a flow,
produce different likelihoods at identical times; each per-stream score records
what its value covered, and an older trace with no such record is refused as
unverifiable rather than passed as agreeing), fits bound to different observed
data (gh#713; override: `--allow-data-mismatch`, which renders the Δ as
confounded), a fit whose training data file no longer hashes to what the fit
recorded when it ran (`compare` re-reads the file, so the row would be scored at
a θ̂ fitted to other bytes; override: `--allow-changed-data`), and traces scored
with different particle or replicate counts, as each trace records them (the log
of a particle-filter likelihood estimate is biased downward, by more at fewer
particles, so part of the Δ would measure the filters; override:
`--allow-mismatched-filter-settings`). Provenance kinds that differ (plug-in
against a posterior mixture) are refused too (override:
`--allow-mixed-provenance`). Under each override that was needed, every format
states it once — the flag, the rows it concerned and what it does to the reading
— as a line under the table and an entry of the JSON's `overrides` array. A
derived row is scored on the backend its fit ran on: a chain-binomial fit by a
`camdl pfilter` child, an ODE fit (`mh` or NLopt on `backend = "ode"`) by its
own ODE likelihood, which gives each observation's exact density at the
trajectory run from the start at θ, averaged over the posterior draws (gh#312;
`basis` in the JSON, and a "Predictive: ODE free-forward" line under the table
of an ODE cohort). A filtered row and an ODE row are refused as a pair, with no
override: the filter's log-likelihood estimate is biased downward and the ODE
score is exact, so their Δ would carry that bias.

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
camdl compare full/fit.toml cases_only/fit.toml --streams cases
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
  (`simulate`, `fit run`, `fit predict`, `pfilter`, `profile`, `survey`,
  `compare` when it derives a prequential, `dev eval`). The error names the
  forcing, its range and the side it was evaluated on.
  `--allow-forcing-extrapolation` on any of them (in a fit config,
  `[config] allow_forcing_extrapolation = true`) holds the forcing's end value
  instead, prints one warning per forcing naming its range and the span held,
  and records the override in each `run.json` the command writes. The override
  is part of the run's identity, so a run under it and the same run without it
  are two runs: neither is served the other's stored result.

- **`simulate` goes forward; `fit` goes backward.** Simulation maps parameters →
  data; fitting maps data → parameters. A sweep is many forward runs, never an
  inference.
- **`survey` and `profile` are about the likelihood _surface_, not a fit.**
  Survey samples it broadly to check identifiability; profile traces it along
  one axis. Both can precede or contextualize a fit; neither produces a
  posterior.
- **`pfilter` is a shortcut into machinery `fit run` also uses.** The same
  particle-filter core backs both surfaces; the standalone command exposes a
  smaller set of knobs for a quick fixed-θ log-likelihood, while a `fit.toml`
  `[method]` adds a start rule, the convergence diagnostics, and post-fit
  audits. An MLE-only run is not a separate command — it is a `fit.toml` whose
  `[method]` is `algorithm = "if2"`, run through `camdl fit run`.
