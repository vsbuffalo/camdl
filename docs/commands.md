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

| Command                  | Produces                                                                       |
| ------------------------ | ------------------------------------------------------------------------------ |
| `simulate` (alias `sim`) | One forward trajectory (+ optional synthetic observations, event log).         |
| `batch run`              | Many trajectories over a grid — see [Scenario sweeps](#scenario-sweeps-batch). |

### Produce inference artifacts

| Command   | Produces                                                                                                                                                                                                             |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit run` | Fit a `fit.toml`'s problem with its one `[method]` — an MLE (`if2`, the NLopt optimizers), a posterior (`pgas`, `pmmh`, `mh`, `nuts`) or a filter evaluation (`pfilter`), with its diagnostics. The production path. |
| `pfilter` | A log-likelihood at _fixed_ parameters via a bootstrap particle filter (no estimation).                                                                                                                              |
| `profile` | A profile-likelihood curve — parallel IF2 over a grid of one focal parameter.                                                                                                                                        |

### Produce diagnostic artifacts

| Command  | Produces                                                                                                                                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `survey` | A likelihood landscape: log-likelihood at many Latin-hypercube points across a parameter box. Answers "is this model identifiable from this data?" _before_ a fit. Not a fitting routine. |

### Read, display, compare

| Command                                            | Does                                                                                                                                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `list`                                             | Browse cached runs as a table.                                                                                                                                                        |
| `show`                                             | Full metadata for one cached run.                                                                                                                                                     |
| `cat`                                              | Emit a cached run's trajectory or observations as TSV.                                                                                                                                |
| `compare`                                          | Paired prequential comparison (elpd, CRPS, PIT) across fits. Scores are plug-in + in-sample-optimistic (the caveat is printed with the table); not a leave-future-out forecast score. |
| `label`                                            | Set a display label on any run.                                                                                                                                                       |
| `fit {run,summary,predict,diff,table,new,methods}` | Run a fit; inspect, summarize, predict-vs-observe, and aggregate fits; scaffold new `fit.toml`s.                                                                                      |
| `batch status`                                     | Completion of a sweep.                                                                                                                                                                |
| `dev eval`                                         | Evaluate model expressions (parameters, forcings) on a time grid — pure inspection, no simulation.                                                                                    |
| `data split`                                       | Split a data TSV into train/holdout.                                                                                                                                                  |
| `lineage {realize,tree,sojourn,cohort}`            | Offline projections over an event log — transmission tree, dwell times, cohort incidence.                                                                                             |

### Compiler passthrough

`check` and `inspect` (and `camdl dev compile`) delegate to `camdlc` — type- and
dimension-check a `.camdl` model, print its compiled structure, and compile it
to IR. `check` reports diagnostics (errors / warnings / lints);
`inspect --summary` prints the structural overview.

## Which methods go through `fit run`

`fit run` is the home for _fitting_. Every estimation and posterior-sampling
method is expressible as a named stage in a `fit.toml`; some also have a
standalone command for quick one-shot use.

| Method                   | `fit.toml` stage (`algorithm = …`) | Standalone command | Role                                                                       |
| ------------------------ | ---------------------------------- | ------------------ | -------------------------------------------------------------------------- |
| IF2 (iterated filtering) | `"if2"`                            | —                  | Maximum-likelihood point estimate.                                         |
| PGAS + NUTS              | `"pgas"`                           | —                  | Bayesian posterior (exact complete-data likelihood, analytical gradients). |
| PMMH                     | `"pmmh"`                           | —                  | Bayesian posterior, gradient-free.                                         |
| Particle filter          | `"pfilter"`                        | `camdl pfilter`    | Log-likelihood evaluation at fixed θ (diagnostic).                         |
| NLopt (Subplex / BOBYQA) | `"nl-sbplx"`, `"nl-bobyqa"`        | —                  | Deterministic MLE on the ODE backend.                                      |
| Survey (LHS landscape)   | —                                  | `camdl survey`     | Identifiability diagnostic; _feeds_ a fit, is not a stage.                 |
| Profile likelihood       | —                                  | `camdl profile`    | Meta-routine running IF2 per grid point.                                   |

Two methods are **fit-run only** — PGAS and PMMH have no standalone command,
because Bayesian sampling presupposes the priors, convergence gates, and staging
that a `fit.toml` provides. Both **require priors** (declared under
`[estimate.<name>.prior]`); IF2, particle filter, and the NLopt optimizers do
not. PGAS uses gradient-based NUTS by default, which requires the compiler to
have emitted `rate_grad` expressions (autodiff); set `use_nuts = false` for a
gradient-free Metropolis-within-Gibbs fallback.

Two routines are **standalone only**. `survey` is a diagnostic, not a fit — it
maps the landscape so you can see ridges and multimodality before committing
compute. `profile` is a meta-fit that runs IF2 at each point of a parameter
grid; it orchestrates fits rather than being a single stage.

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
`(time, <dims>)` and plot, one facet per stratum. Omit `--horizon` for all
applicable horizons (chain-binomial → `free_forward` + `one_step`; ODE →
`free_forward` only); an optimizer fit (IF2 / NLopt) is refused since it has no
posterior cloud.

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

### Scenario sweeps: batch

`batch run` is forward simulation only — **no inference**. It runs `simulate`
across the Cartesian product of a parameter sweep (or a space-filling design) ×
scenarios × seed replicates, with optional synthetic-observation sampling:

```toml
[config]
model = "model.camdl"
backend = "chain_binomial"

[[scenario]]
name = "intervention"
enable = ["school_closure"]

[sweep]
beta = { linspace = [0.2, 0.6, 5] }

[design.sensitivity] # alternative to a grid: LHS / Sobol / random
method = "lhs"
n = 200
```

```bash
camdl batch run sweep.toml
camdl batch run sweep.toml --dry-run           # preview the resolved grid
camdl batch status sweep.toml
```

Use `batch` to explore _forward_ behavior under interventions and parameter
ranges. To estimate parameters from data, use `fit`.

### Model comparison

```bash
camdl compare results/fits/a/posterior results/fits/b/posterior --baseline a
camdl compare @baseline @candidate
camdl compare @baseline @candidate --pointwise pointwise.tsv
```

`compare` renders a baseline-centered table of prequential scores (per-step
log-score, CRPS, PIT). Each argument is either an explicit prequential artifact
— a `prequential.json` (or a stage dir holding one), written by a `pfilter`
stage within a fit or by `camdl pfilter --save-prequential`, and read **as-is**
— **or** a fit handle (`@label`, a hash prefix, a run directory, or a
`fit.toml`), whose prequential is **auto-derived** by re-filtering at the fit's
sealed θ̂. `--particles` and `--seed` set the filter used for any auto-derived
handle and are applied **uniformly** across all derived fits, so `T_score` and
the scores stay commensurable; they are ignored for an explicit
`prequential.json` (read as-is). The scores are **plug-in and
in-sample-optimistic** — computed at a single θ fit to the whole series — so
they are not a leave-future-out forecast score. The optimism does not cancel
when two models are differenced, either: it grows with the effective number of
parameters each model was free to tune against the same observations, so Δelpd
tilts toward the more flexible model. Read a comparison as indicative, alongside
`se(Δ)` and the caveats.

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

`compare` refuses, rather than renders, five unlike-for-unlike comparisons: a
differing `T_score` (override: `--allow-mismatched-horizon`, which suppresses
the Δ columns), traces scored at different observation times, traces that scored
different observation streams at the same time (gh#570 — the step's log score is
the joint score over those streams), traces that scored the same stream at the
same time over different _windows_ (gh#833 — a gapped reading of a file against
a merged one, or a stream read as an instant against the same name read as a
flow, produce different likelihoods at identical times; each per-stream score
records what its value covered, and an older trace with no such record is
refused as unverifiable rather than passed as agreeing), and fits bound to
different observed data (gh#713; override: `--allow-data-mismatch`, which
renders the Δ as confounded). A fit whose terminal stage ran on a backend other
than `chain_binomial` is refused on the derive path, since the derived score
would come from a different forward process than the fit used (gh#312).

`--pointwise PATH` writes the per-observation difference the table already
computes in order to form `se(Δelpd)`, as a TSV with one row per candidate ×
scored step, joint and per stream: `model`, `baseline`, `t`, `scope`, `stream`,
`log_score`, `baseline_log_score`, `delta_log_score`. `Δelpd = 12 nats` says a
model won; this says _where_ it won — on three weeks around an intervention, on
one district, on a single reporting batch. An elpd gap taken across two
different stream sets never reaches this file: the preflight refuses that
comparison before the table is rendered or the TSV is written, naming the
streams that differ (gh#570).

## The boundaries, stated plainly

- **`simulate` and `batch` go forward; `fit` goes backward.** Simulation maps
  parameters → data; fitting maps data → parameters. Batch is many forward runs,
  never an inference.
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
