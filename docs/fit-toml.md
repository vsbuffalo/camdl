# The `fit.toml` reference

A `fit.toml` is the runtime config for `camdl fit run` — it names the model, the
data, what to estimate, what to fix, and the inference stages. It is **not**
part of the model language: the `.camdl` file declares parameter names, bounds,
and priors; the `fit.toml` selects which parameters to estimate and how. Every
field below is verified against the parser (`config_v2.rs`).

For the workflow these configs drive, see `camdl docs workflow`.

## A complete example

```toml
output_dir = "results"            # where runs are stored (optional; relative
                                  # to this file — see "How paths resolve")

[model]
camdl = "model.camdl"

[data.observations]               # one key per observation stream in the model
cases = "data/cases.tsv"

[estimate]                        # the parameters to infer
beta  = { bounds = [0.001, 0.5], start = 0.04, transform = "log",
          prior = { log_normal = { mu = -2.0, sigma = 1.0 } } }
gamma = { bounds = [0.01,  1.0], start = 0.12,
          prior = { log_normal = { mu = -1.2, sigma = 0.5 } } }
s0    = { bounds = [0.01, 0.30], perturb_only_at_t0 = true }  # initial state

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
model, each mapped to a TSV path.

`[data.holdout]` (same shape) and `holdout_after = <time>` (under `[data]`)
parse, are validated, and are folded into the fit's identity — but **the split
is not yet applied** (gh#585). Every fit trains on all the bound observations,
and no held-out score is computed, so a score from such a fit is in-sample no
matter which of these keys is set. Do not report one as out-of-sample. For an
honest held-out number today, fit on a truncated data file and then score the
**full** series with `camdl pfilter --save-prequential`, summing the `log_score`
of the `joint` rows past the split time — scoring the held-out file on its own
starts the filter from the prior with nothing assimilated from the training
window, which is a different (and unfairly harsh) quantity. The recipe is
spelled out under "Out-of-sample validation" in `camdl docs inference`.

**`[estimate]`** — the parameters to infer. Each value is an inline table:

| key                                          | meaning                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bounds = [lo, hi]`                          | search range. _Optional_ — defaults to the model's `parameters { p : rate in [lo,hi] }` range; a `fit.toml` `bounds` may only **narrow** it, never loosen.                                                                                                                                                        |
| `start = X`                                  | the base starting value. Optional — defaults to the model's declared value, else a draw from bounds. See "Where a chain starts" below.                                                                                                                                                                            |
| `prior = { … }`                              | prior distribution. **Required** for a `pgas`/`pmmh` stage; `if2` ignores it. See "Priors" below.                                                                                                                                                                                                                 |
| `transform = "log" \| "logit" \| "identity"` | inference-scale transform. Optional — inferred from the parameter's declared type if omitted.                                                                                                                                                                                                                     |
| `perturb_only_at_t0 = true`                  | an initial-state parameter (e.g. `s0`, `i0`) — perturbed at t=0 only, never at an observation. It is an IF2 perturbation schedule: `if2` reads it and every other stage ignores it, which is fine. A config-load error only when the fit has **no** `if2` stage at all, since there the declaration does nothing. |
| `rw_sd = X`                                  | IF2 per-parameter random-walk SD. Optional — auto-scaled from bounds.                                                                                                                                                                                                                                             |

**`[fixed]`** — `param = value` for every model parameter you are _not_
estimating. camdl requires every declared parameter to be either estimated or
fixed.

**`[stages.<name>]`** — stages are **user-named**, and the order they appear in
the file is the order they run. `algorithm` picks the method and `backend` the
simulator it fits against: `chain_binomial` for the stochastic-process methods
(`if2`, `pgas`, `pmmh`, `pfilter` — the ones that need chain-binomial process
noise and `balance`), or `ode` for the deterministic-likelihood methods
(`nl-sbplx` / `nl-bobyqa` MLE and the Bayesian `mh` / `nuts`). Each algorithm is
valid on exactly one backend — `camdl fit methods` lists the pairs, and an
invalid pair is rejected at load. A downstream stage warm-starts from an
upstream one with `init_mle = "<stage-name>"`.

**`[config]`** — fit-wide simulator settings: `dt` (the integrator step, default
`1.0`). The `dt` you care about lives here; a `dt` written at the top level of
the file is a typo, not a setting. (The forward backend for synthetic-data
generation is `[synthetic].backend`, not a `[config]` setting — gh#241; the fit
stages declare their own `backend`.)

## How paths resolve

**Every path written in the `fit.toml` is relative to the `fit.toml` itself** —
`[model].camdl`, `[data].file`, each `[data.observations]` and `[data.holdout]`
stream, and `output_dir`. Absolute paths pass through unchanged (and draw a
portability warning, since they pin the config to one machine's layout).

The rule is the one Cargo and `pyproject.toml` use: a path written **in a file**
anchors at that file, so the config is relocatable as a unit and runs the same
from any working directory. What does _not_ come from the file — the `results/`
default when no `output_dir` is declared, and `CAMDL_OUTPUT_DIR` — anchors at
the working directory instead, since that is the frame you typed it in.

Concretely, for this layout:

```
<repo>/camdl/fit.toml
<repo>/camdl/bvd_province.camdl
<repo>/data/build/camdl/cases.tsv
```

every path takes the same base, whichever directory you run from:

```toml
output_dir = "../results" # <repo>/results

[model]
camdl = "bvd_province.camdl" # <repo>/camdl/bvd_province.camdl

[data.observations]
cases = "../data/build/camdl/cases.tsv" # <repo>/data/build/camdl/cases.tsv
```

`fit run` prints the resolved output location as an **absolute** path at start,
so you can confirm where the run tree is going before it is written.

## Where a chain starts

Two settings decide this, and they answer different questions.

**`[estimate].start` sets the base point** — one θ for the whole stage. It is
the top of a precedence chain: an upstream stage's result (`init_mle`) beats it,
and below it sit `[fixed]`, the model's declared parameter value, and finally a
draw from bounds if nothing else supplies one.

**`init` decides how the chains are spread around that base point.** A stage
with `chains = 1`, or with `init = "single"`, has nothing to spread, so every
chain runs from the base point itself.

| `init`                                        | where the chains start                                                                                                                                         |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `single`                                      | every chain at the base point                                                                                                                                  |
| `uniform`                                     | chain 1 at the base point; the rest uniform within `bounds`                                                                                                    |
| `lhs`                                         | Latin-hypercube stratified over `bounds`; base point unused                                                                                                    |
| `uniform_unconstrained` (default)             | spread on the unconstrained scale; base point unused                                                                                                           |
| `survey_top_k`                                | the top-K points from a `camdl survey` run; base point used only for a parameter the survey did not sweep                                                      |
| `from_prior`                                  | draws from the declared priors; base point unused                                                                                                              |
| `from_posterior` / `from_mle` / `from_params` | rows or values read from the named source; a name missing from that source falls back to bounds-uniform, or to the base point when the model declares no range |

The three spreading modes — `uniform`, `lhs`, `uniform_unconstrained` — fall
back to the base point at `chains = 1`, since with one chain there is nothing to
spread. The source-reading modes do not: they read one row from their source
however many chains you asked for.

So `start` is load-bearing wherever the table says "base point". When a mode
ignores it, that is deliberate: the mode's whole purpose is to explore, and the
chain-agreement gate is only informative if the chains genuinely start apart.

Each stage writes a `chain_starts.tsv` recording where every chain actually
began, before any perturbation. That file, not the config, is the authority on
what a run did.

> **Unknown keys are rejected.** A misplaced or misspelled key is a hard error
> naming the offending key — `fit.toml` is parsed strictly. A top-level `dt` (it
> belongs in `[config]`) or `particle` (it is `particles`, under a stage) fails
> at load rather than being silently dropped, so a sweep that varies a typo'd
> knob can't quietly produce identical fits.

## Conditioning boundary (`condition_from`)

### Why it exists

An **incidence** observation (a weekly case count, say) is the flow accumulated
over one reporting interval — `(t_{k-1}, t_k]`. The _first_ observation is the
only one whose left edge isn't a previous observation; by default the filter
opens it at the model origin (`simulate.from` / `t_start`), so `y_1` is scored
against every event since the dynamics began. That is correct when the data
starts about one cadence after `t_start`. It is **wrong** when `t_start` sits
far behind the first datum — e.g. you start dynamics in 2011 so births and
SIA/MCV covariates shape the susceptible pool, but case data begins in 2014. The
first bin then spans the whole 2011–2014 warm-up, and scoring one weekly count
against three years of flow is meaningless (gh#134) — and it collapses the
particle filter (no particle's three-year integral matches a single datum).

The fix is a **conditioning boundary**: the leading span `[t_start, cond_from)`
becomes a covariate-informed **warm-up** — simulated with the full stochastic
dynamics (births, campaigns, seasonality, process noise) but **not scored** —
and the first observation is scored against one normal cadence
`(cond_from, first_obs]`. Mechanically it is a leading reset-only point on the
stream's grid: the incidence accumulator resets at `cond_from`, discarding the
warm-up flow, with no likelihood term there.

### Conditioning is explicit — you state it, the filter never guesses

camdl does **not** infer the boundary. An inferred boundary would be fragile (it
fails exactly on the irregular/sparse surveillance data this is for) and fail
_silently_. So you set it, and it is **required** precisely when it matters: an
**incidence** stream whose first observation lands anomalously far after
`t_start` (a wide leading window relative to that stream's own cadence) with no
`condition_from` is a **hard error (W329)** that names the fix. A stream whose
first observation is ~one cadence after `t_start` (the common, well-set-up case)
needs nothing. A **prevalence** stream is exempt (its `y_1` reads the state at
the instant, not a flow integral) — a wide gap there is only a soft warning.

### The surface — one default, optional per-stream shadows

`condition_from` is a top-level key with two forms:

```toml
# (1) a string — the default applied to EVERY stream:
condition_from = "first_obs - 1 week" # one cadence before the data
# condition_from = "date(\"2014-08-18\")"  # an absolute calendar date
# condition_from = "19"                     # an absolute model-time number (quoted)
```

```toml
# (2) a table — an optional all-streams `default` plus per-observation-label
#     SHADOWS, for multi-cadence models (streams on different schedules):
[condition_from]
default = "first_obs - 1 week" # applied to every stream …
es = "first_obs - 2 weeks" # … except `es`, which this shadows
afp = "first_obs - 1 month" # … and `afp`
```

**Resolution per stream:** its shadow → else the `default` → else _none_ (and
then W329 decides whether none is fine or a hard error). The shadow key is the
observation-block label (the `[data]` key). `default` is reserved — a stream
literally labelled `default` is a hard error; so is a shadow naming a stream
that doesn't exist (typo-safety, the error lists the valid labels).

A spec must resolve to a time strictly between `t_start` and that stream's first
observation (and onto the `dt` grid). The duration form (`first_obs - 1 week`)
is anchored to **each stream's own** first observation, so in a multi-cadence
model the same `default` gives each stream a window in its own cadence.

`condition_from` and `ic_free` cannot be combined (the leading hole would leave
`y_1` with nothing to condition on).

`camdl pfilter` and `camdl profile` apply the same conditioning, so a fixed-θ
loglik is computed over the same scored window as the fit's: they read the
`--fit` toml's `condition_from`, overridden by the repeatable flag
`--condition-from SPEC` (all-streams default) / `--condition-from LABEL=SPEC`
(per-stream shadow). The W329 wide-first-window hard error applies there
identically.

`camdl fit predict` carries the same window into both predictive horizons, so a
predicted row and the observed row it is plotted against cover the same
interval: the free-forward projection opens the first incidence bin at
`condition_from`, and the one-step filter is handed the same leading reset. The
boundary is a reset, not an observation — no predictive row is emitted at it.
The free-forward projection reads the recorded cumulative flow at the boundary,
so `condition_from` must also be a **recorded output time**; if it is not,
`fit predict` refuses and names the fix (widen `output { trajectories { … } }`,
or move the boundary onto an output time).

## Priors

Externally-tagged inline tables (the wire format matches the IR emission):

```toml
prior = { log_normal = { mu = 0.0, sigma = 1.0 } }
prior = { normal = { mean = 0.0, sd = 1.0 } }
prior = { beta = { alpha = 2.0, beta = 5.0 } }
prior = { log_uniform = { lower = 1e-5, upper = 1e-2 } } # uniform on the log scale
prior = { truncated_normal = { mean = 0.7, sd = 0.2, lower = 0.3, upper = 1.0 } }
prior = { uniform = { lower = 0.0, upper = 1.0 } } # explicit bounds
prior = { uniform = {} } # uniform over the param's `bounds`
prior = { half_normal = { sigma = 1.0 } }
prior = { flat = {} } # explicit "flat on purpose" — only valid in fit.toml
```

The empty `uniform = {}` form is uniform over the parameter's `bounds` (the
`[estimate.<name>].bounds`, falling back to the model's `in [lo, hi]`) — a
convenience so you don't repeat the interval. It requires bounds from one of
those sources. `truncated_normal`'s `lower`/`upper` must equal the parameter's
bounds (the prior's support and the search box are the same interval).

**Precedence:** a `fit.toml` `[estimate].prior` overrides the model's `~`
declaration; if neither is present, a Bayesian stage falls back to flat **with a
warning** — camdl refuses _silent_ implicit-flat priors, because the prior shows
up in the posterior. The explicit `{ flat = {} }` is how you say "flat here, on
purpose" without the warning.

## Stage algorithms

| `algorithm`              | backend          | role                                              | key fields                                                                            |
| ------------------------ | ---------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `if2`                    | `chain_binomial` | iterated filtering → MLE                          | `chains`, `particles`, `iterations`, `cooling` (+ `cooling_target_iters`, default 50) |
| `pgas`                   | `chain_binomial` | particle Gibbs + NUTS → posterior                 | `chains`, `particles`, `sweeps` (+ `burn_in`, `thin`, `tempering`, `max_tree_depth`)  |
| `pmmh`                   | `chain_binomial` | particle marginal MH → posterior                  | `chains`, `particles`, `iterations`                                                   |
| `pfilter`                | `chain_binomial` | particle filter at fixed θ → log-likelihood + ESS | `particles`, `replicates`                                                             |
| `nl-sbplx` / `nl-bobyqa` | `ode`            | NLopt deterministic optimizer → MLE               | `chains` (LHS starts) (+ `max_evals`, `tolerance`)                                    |
| `mh`                     | `ode`            | MH on the deterministic ODE marginal → posterior  | `chains`, `iterations` (+ `burn_in`, `thin`, `adapt`, `adapt_start`)                  |
| `nuts`                   | `ode`            | gradient NUTS (forward sensitivities) → posterior | `chains`, `warmup`, `samples` (+ `max_tree_depth`, `target_accept`, `dense_mass`)     |

The `ode`-backend Bayesian samplers (`mh`, `nuts`) fit the **deterministic
marginal likelihood** `p(y | θ, ODE skeleton)` rather than the stochastic
`p(y | θ)` — a different statistical object, appropriate for equilibrium or
large-population models. `nuts` requires a differentiable model (the capability
gate refuses an undifferentiable gradient, an adaptive `rk45` integrator, a
scheduled effect, or an initial condition the gradient path cannot seed); `mh`
is gradient-free and carries no such requirement. See `camdl docs inference`
(the ODE-backend fitting section) for when to pick which.

```toml
# A gradient-based Bayesian fit on the ODE skeleton.
[stages.posterior]
algorithm = "nuts" # or "mh" for the gradient-free sampler (`iterations` + `burn_in`)
backend = "ode"
chains = 4
warmup = 500 # step-size adaptation draws (discarded)
samples = 500 # posterior draws kept per chain
```

Common to every stage:

- `init = "uniform_unconstrained"` (default, Stan-style boundary-avoiding draws
  on the unconstrained scale) `| "lhs" | "single" | "uniform" | "survey_top_k"`
  — how per-chain starting points are drawn.
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
