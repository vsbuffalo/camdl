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

| Command   | Produces                                                                                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fit run` | The full inference pipeline: a `fit.toml`'s named stages, in order — MLE, posterior, diagnostics. The production path. An MLE-only fit is a `fit.toml` with one `algorithm = "if2"` stage. |
| `pfilter` | A log-likelihood at _fixed_ parameters via a bootstrap particle filter (no estimation).                                                                                                    |
| `profile` | A profile-likelihood curve — parallel IF2 over a grid of one focal parameter.                                                                                                              |

### Produce diagnostic artifacts

| Command  | Produces                                                                                                                                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `survey` | A likelihood landscape: log-likelihood at many Latin-hypercube points across a parameter box. Answers "is this model identifiable from this data?" _before_ a fit. Not a fitting routine. |

### Read, display, compare

| Command                                            | Does                                                                                                                                                            |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `list`                                             | Browse cached runs as a table.                                                                                                                                  |
| `show`                                             | Full metadata for one cached run.                                                                                                                               |
| `cat`                                              | Emit a cached run's trajectory or observations as TSV.                                                                                                          |
| `compare`                                          | Paired prequential comparison (elpd, CRPS, PIT) across fits. Scores are plug-in + in-sample-optimistic (caveat printed); not a leave-future-out forecast score. |
| `label`                                            | Set a display label on any run.                                                                                                                                 |
| `fit {run,summary,predict,diff,table,new,methods}` | Run a fit; inspect, summarize, predict-vs-observe, and aggregate fits; scaffold new `fit.toml`s.                                                                |
| `batch status`                                     | Completion of a sweep.                                                                                                                                          |
| `dev eval`                                         | Evaluate model expressions (parameters, forcings) on a time grid — pure inspection, no simulation.                                                              |
| `data split`                                       | Split a data TSV into train/holdout.                                                                                                                            |
| `lineage {realize,tree,sojourn,cohort}`            | Offline projections over an event log — transmission tree, dwell times, cohort incidence.                                                                       |

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

A fit stage can then **start from the survey's best regions** rather than from
random points, by reading the top-K landscape rows:

```toml
[stages.scout]
algorithm = "if2"
init = "survey_top_k"
survey_path = "results/surveys/model-abc123/"
survey_top_k_n = 10
```

The survey is consumed as a starting-point source; it never becomes a stage.

### The fit pipeline: scout → refine → posterior

A `fit.toml` declares named stages that run in order. The canonical ladder is a
fast MLE _scout_ (IF2) to locate the mode, a _refine_ that sharpens it, and a
_posterior_ stage (PGAS) that characterizes uncertainty:

```toml
[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 10
particles = 500
iterations = 50

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 1000
sweeps = 1000
init_mle = "scout" # start from scout's MLE
init = "single" # all chains at that point
```

```bash
camdl fit run fit.toml                              # all stages, in order
camdl fit run fit.toml --stage scout                # one stage only
camdl fit run fit.toml --resume <base-run-id> --stage posterior   # extend a completed run
camdl fit summary results/fits/<dir>/               # Â / gate verdict / MLE table
camdl fit predict --fit fit.toml --stream onset     # predicted-vs-observed artifact
```

`fit predict` resolves the fit's posterior draws and writes
`predictive/<stream>.tsv` (the `q05…q95` ribbon, with typed
`horizon`/`treatment`/`rhat_max` columns) and `observed/<stream>.tsv` under the
run directory. Join the two on `(time, <dims>)` and plot, one facet per stratum.
Omit `--horizon` for all applicable horizons (chain-binomial → `free_forward` +
`one_step`; ODE → `free_forward` only); an optimizer fit (IF2 / NLopt) is
refused since it has no posterior cloud.

A resumed fit reads the base run read-only and writes a _new_ run keyed on the
extended length. It is a distinct deterministic artifact — not bit-identical to
an uninterrupted fit of the same length (both are valid posterior samples).

**Initialization and staging.** Each stage, on completion, records its best
estimate (θ̂). A downstream stage picks where to start with two knobs:

- **the source** — where the starting point comes from: random bounds, a prior,
  an _upstream stage's_ MLE or posterior, an explicit params file, or a survey's
  top-K rows;
- **the spread** — how chains are distributed around it: Stan-style
  boundary-avoiding draws on the unconstrained scale (`uniform_unconstrained`,
  the default), all at one point (`single`), Latin-hypercube perturbation
  (`lhs`), or uniform over bounds.

Both PGAS and PMMH can start from an IF2 scout's MLE this way — there is no
asymmetry between them. A posterior stage can equally start `from_posterior`
(another PGAS stage), `from_params` (a file), or `from_prior`.

Stages are chained by a **convergence gate**: before a downstream stage runs,
the runner checks that its upstream dependency actually converged (tail
chain-agreement Â), and refuses to proceed otherwise. A poor scout blocks the
refine rather than silently seeding it with a bad mode.

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
they support _relative_ comparison but are not a leave-future-out forecast
score; `compare` prints this caveat on every run.

`--pointwise PATH` writes the per-observation difference the table already
computes in order to form `se(Δelpd)`, as a TSV with one row per candidate ×
scored step, joint and per stream: `model`, `baseline`, `t`, `scope`, `stream`,
`log_score`, `baseline_log_score`, `delta_log_score`. `Δelpd = 12 nats` says a
model won; this says _where_ it won — on three weeks around an intervention, on
one district, on a single reporting batch. A stream only one side scored gets an
empty difference rather than a number, so an elpd gap taken across two different
stream sets is visible instead of hidden inside the scalar.

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
  smaller set of knobs for a quick fixed-θ log-likelihood, while `fit.toml`
  stages add initialization sources, convergence gates, and post-fit audits. An
  MLE-only run is not a separate command — it is a `fit.toml` with a single
  `algorithm = "if2"` stage, run through `camdl fit run`.
