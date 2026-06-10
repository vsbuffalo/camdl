# The `fit.toml` reference

A `fit.toml` is the runtime config for `camdl fit run` — it names the model, the
data, what to estimate, what to fix, and the inference stages. It is **not**
part of the model language: the `.camdl` file declares parameter names, bounds,
and priors; the `fit.toml` selects which parameters to estimate and how. Every
field below is verified against the parser (`config_v2.rs`).

For the workflow these configs drive, see `camdl docs workflow`.

## A complete example

```toml
output_dir = "results"            # where runs are stored (optional)

[model]
camdl = "model.camdl"

[data.observations]               # one key per observation stream in the model
cases = "data/cases.tsv"

[estimate]                        # the parameters to infer
beta  = { bounds = [0.001, 0.5], start = 0.04, transform = "log",
          prior = { log_normal = { mu = -2.0, sigma = 1.0 } } }
gamma = { bounds = [0.01,  1.0], start = 0.12,
          prior = { log_normal = { mu = -1.2, sigma = 0.5 } } }
s0    = { bounds = [0.01, 0.30], ivp = true }        # initial-value parameter

[fixed]                           # held at a value, not estimated
rho = 0.6
k   = 10.0

[stages.scout]                    # find the basin (MLE)
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 8
particles  = 2000
iterations = 150
cooling    = 0.7

[stages.posterior]                # sample the posterior, warm-started from scout
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 4
particles = 600
sweeps    = 300
init_mle  = "scout"               # take this stage's base point from the scout stage
```

## Sections

**`[model]`** — `camdl = "path/to/model.camdl"`.

**`[data.observations]`** — one key per observation stream declared in the
model, each mapped to a TSV path. For out-of-sample validation, add `[holdout]`
(same shape) or `holdout_after = <time>`.

**`[estimate]`** — the parameters to infer. Each value is an inline table:

| key                                          | meaning                                                                                                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bounds = [lo, hi]`                          | search range. _Optional_ — defaults to the model's `parameters { p : rate in [lo,hi] }` range; a `fit.toml` `bounds` may only **narrow** it, never loosen. |
| `start = X`                                  | starting value. Optional — random from bounds (scout) or inherited from an upstream stage.                                                                 |
| `prior = { … }`                              | prior distribution. **Required** for a `pgas`/`pmmh` stage; `if2` ignores it. See "Priors" below.                                                          |
| `transform = "log" \| "logit" \| "identity"` | inference-scale transform. Optional — inferred from the parameter's declared type if omitted.                                                              |
| `ivp = true`                                 | initial-value parameter (e.g. `s0`, `i0`) — perturbed only at t=0.                                                                                         |
| `rw_sd = X`                                  | IF2 per-parameter random-walk SD. Optional — auto-scaled from bounds.                                                                                      |

**`[fixed]`** — `param = value` for every model parameter you are _not_
estimating. camdl requires every declared parameter to be either estimated or
fixed.

**`[stages.<name>]`** — stages are **user-named**, and the order they appear in
the file is the order they run. `algorithm` picks the method; `backend` the
simulator (`chain_binomial` for fits — needed for chain-binomial process noise
and `balance`). A downstream stage warm-starts from an upstream one with
`init_mle = "<stage-name>"`.

**`[config]`** — fit-wide simulator settings: `backend` (default
`chain_binomial`) and `dt` (the integrator step, default `1.0`). The `dt` you
care about lives here; a `dt` written at the top level of the file is a typo,
not a setting.

> **Unknown keys are rejected.** A misplaced or misspelled key is a hard error
> naming the offending key — `fit.toml` is parsed strictly. A top-level `dt` (it
> belongs in `[config]`) or `particle` (it is `particles`, under a stage) fails
> at load rather than being silently dropped, so a sweep that varies a typo'd
> knob can't quietly produce identical fits.

## Conditioning boundary (`condition_from`)

A top-level key. By default the filter conditions from the model origin
(`simulate.from`): the first observation is scored against the flow accumulated
since `t_start`. When `simulate.from` sits well before the first datum — e.g.
you start dynamics in 2011 so births and SIA/MCV covariates shape the
susceptible pool, but case data begins in 2014 — that first incidence window
spans the whole gap, and the opening likelihood term is meaningless (gh#134).

`condition_from` moves the conditioning boundary to one cadence before the first
datum. The leading span becomes a covariate-informed **warm-up** — simulated
with the full stochastic dynamics (births, campaigns, seasonality, process
noise) but **not scored** — and the first observation is scored against one
normal cadence:

```toml
condition_from = "first_obs - 1 week" # idiomatic: one cadence before the data
# condition_from = "date(\"2014-08-18\")"  # or an absolute calendar date
```

It must resolve to a time strictly between `simulate.from` and the first
observation (and onto the `dt` grid). For an **incidence** stream, omitting it
when the leading gap is large is a hard error (W329) naming this fix; for
**prevalence** it is only a soft warning. `condition_from` and `ic_free` cannot
be combined.

## Priors

Externally-tagged inline tables (the wire format matches the IR emission):

```toml
prior = { log_normal = { mu = 0.0, sigma = 1.0 } }
prior = { normal = { mean = 0.0, sd = 1.0 } }
prior = { beta = { alpha = 2.0, beta = 5.0 } }
prior = { uniform = {} } # uniform over bounds
prior = { half_normal = { sigma = 1.0 } }
prior = { flat = {} } # explicit "flat on purpose" — only valid in fit.toml
```

**Precedence:** a `fit.toml` `[estimate].prior` overrides the model's `~`
declaration; if neither is present, a Bayesian stage falls back to flat **with a
warning** — camdl refuses _silent_ implicit-flat priors, because the prior shows
up in the posterior. The explicit `{ flat = {} }` is how you say "flat here, on
purpose" without the warning.

## Stage algorithms

| `algorithm` | role                                              | key fields                                                                            |
| ----------- | ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `if2`       | iterated filtering → MLE                          | `chains`, `particles`, `iterations`, `cooling` (+ `cooling_target_iters`, default 50) |
| `pgas`      | particle Gibbs + NUTS → posterior                 | `chains`, `particles`, `sweeps` (+ `burn_in`, `thin`, `tempering`, `max_tree_depth`)  |
| `pmmh`      | particle marginal MH → posterior (experimental)   | `chains`, `particles`, `iterations`                                                   |
| `pfilter`   | particle filter at fixed θ → log-likelihood + ESS | `particles`, `replicates`                                                             |

Common to every stage:

- `init = "lhs"` (default, scale-aware Latin-hypercube)
  `| "single" | "uniform" | "survey_top_k"` — how per-chain starting points are
  drawn.
- `init_mle = "<upstream-stage>"` — where this stage's base point comes from.

### Seeding chains from a survey

The survey → fit handoff (workflow step 3 → 4):

```toml
[stages.scout]
algorithm = "if2"
init = "survey_top_k" # draw chain starts from a survey landscape
survey_path = "results/surveys/<survey-run-dir>"
# survey_top_k_n defaults to `chains`
```

### Tempering (PGAS)

```toml
tempering = [
  1.0,
  0.7,
  0.4,
  0.15,
] # first entry MUST be 1.0 (cold chain); only the cold rung samples
```

Add intermediate β values when the tempering swap rate is low (see the
diagnostics table in `camdl docs workflow`).

## How `fit.toml` relates to the model

The model file is the source of truth for what _can_ be estimated; the
`fit.toml` chooses and configures. Specifically:

- **Bounds** default to the model's declared range; `fit.toml` can only narrow.
- **Priors** default to the model's `~` declarations; `fit.toml` overrides.
- **Transforms** default to the parameter's declared type; `fit.toml` overrides.

So the minimal `fit.toml` for a model that already declares bounds and priors is
just `[model]`, `[data]`, an `[estimate]` listing names (no per-param fields),
`[fixed]`, and the stages.
