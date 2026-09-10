# The `fit.toml` reference

A `fit.toml` is the runtime config for `camdl fit run` — it names the model, the
data, what to estimate, what to fix, and the one inference method to fit with.
It is **not** part of the model language: the `.camdl` file declares parameter
names, bounds, and priors; the `fit.toml` selects which parameters to estimate
and how. Every field below is verified against the parser (`config_v2.rs`).

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

[method]                          # the one way this file fits the problem
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 4
particles = 600
sweeps    = 300
# starts = "from_prior"           # the default here: every parameter has a prior
```

The file has two halves. Everything above `[method]` is the **problem** — the
model, the data, the estimate/fixed partition, the priors — and every reader of
a fit config reads it: `simulate --draws prior --fit`, `pfilter --fit`,
`survey --fit`, `profile --fit`. A file with no `[method]` at all is a complete
problem for those. `[method]` is the **inference**: how `fit run` fits that
problem. A second way of fitting the same problem is a second file with the same
problem half and a different `[method]`
(`camdl fit new --from fit.toml
fit-if2.toml` copies the file with provenance;
edit the table). The store puts both under one `fits/<stem>-<h8>/` segment,
because that level hashes the problem alone.

## Sections

**`[model]`** — `camdl = "path/to/model.camdl"`.

**`[data.observations]`** — one key per observation stream declared in the
model, each mapped to a TSV path.

`[data.holdout]` (same shape) and `holdout_after = <time>` (under `[data]`) are
**applied at fit load** (gh#585): `holdout_after` truncates training to
`t ≤ <time>` (it accepts a model-time number, a date under a calendar-anchored
model, or `last_obs - 6 weeks`); `[data.holdout]` files must lie strictly after
each stream's last training time (tail-only). The applied training window is
recorded in `fit.meta.json`, and `camdl compare` scores such fits held-out by
default (`--in-sample` opts out). For a fit that predates the split being
applied, score the **full** series with
`camdl pfilter --save-prequential --score-from <time>`, summing the `log_score`
of the `joint` rows past the split time — scoring the held-out file on its own
starts the filter from the prior with nothing assimilated from the training
window, which is a different (and unfairly harsh) quantity. The recipe is
spelled out under "Out-of-sample validation" in `camdl docs inference`.

`[synthetic]` may stand beside `[data]`: `fit run` then fits the real data, and
the synthetic block records the truth a recovery check reads.

**`[estimate]`** — the parameters to infer. Each value is an inline table:

| key                                          | meaning                                                                                                                                                                                                                     |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bounds = [lo, hi]`                          | search range. _Optional_ — defaults to the model's `parameters { p : rate in [lo,hi] }` range; a `fit.toml` `bounds` may only **narrow** it, never loosen.                                                                  |
| `start = X`                                  | the base starting value. Optional — defaults to the model's declared value, else a draw from bounds. See "Where a chain starts" below.                                                                                      |
| `prior = { … }`                              | prior distribution. **Required** for a `pgas`/`pmmh`/`mh`/`nuts` method; `if2` ignores it. See "Priors" below.                                                                                                              |
| `transform = "log" \| "logit" \| "identity"` | inference-scale transform. Optional — inferred from the parameter's declared type if omitted.                                                                                                                               |
| `perturb_only_at_t0 = true`                  | an initial-state parameter (e.g. `s0`, `i0`) — perturbed at t=0 only, never at an observation. It is an IF2 perturbation schedule: `if2` reads it and every other method ignores it, so a shared problem half can carry it. |
| `rw_sd = X`                                  | IF2 per-parameter random-walk SD. Optional — auto-scaled from bounds. Ignored by every other method.                                                                                                                        |

**`[fixed]`** — `param = value` for every model parameter you are _not_
estimating. camdl requires every declared parameter to be either estimated or
fixed.

**`[method]`** — the one inference method. `algorithm` picks the method and
`backend` the simulator it fits against: `chain_binomial` for the
stochastic-process methods (`if2`, `pgas`, `pmmh`, `pfilter` — the ones that
need chain-binomial process noise and `balance`), or `ode` for the
deterministic-likelihood methods (`nl-sbplx` / `nl-bobyqa` MLE and the Bayesian
`mh` / `nuts`). Each algorithm is valid on exactly one backend —
`camdl fit methods` lists the pairs, and an invalid pair is rejected at load.
`starts` says where the chains begin (below). There is no map of named stages
and no execution order: a file declares one method, and a pipeline is two files
run in turn.

**`[config]`** — fit-wide simulator settings: `dt` (the integrator step, default
`1.0`). The `dt` you care about lives here; a `dt` written at the top level of
the file is a typo, not a setting. (The forward backend for synthetic-data
generation is `[synthetic].backend`, not a `[config]` setting — gh#241; the
method declares its own `backend`.)

## How paths resolve

**Every path written in the `fit.toml` is relative to the `fit.toml` itself** —
`[model].camdl`, `[data].file`, each `[data.observations]` and `[data.holdout]`
stream, `output_dir`, and a `starts` source that is a file
(`from_params = "theta.toml"`). Absolute paths pass through unchanged (and draw
a portability warning, since they pin the config to one machine's layout).

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

`starts`, under `[method]`, is one key for one concept: where the chains begin.
Its value is a bare rule name, or a one-key table naming a rule and its source.

```toml
[method]
starts = "from_prior" # one independent draw per chain
starts = "uniform_unconstrained"
starts = "lhs"
starts = "uniform"
starts = "single" # every chain at the base point
starts = { from_posterior = "@base" } # one row of @base's posterior per chain
starts = { from_mle = "@mle" } # every chain at @mle's point estimate
starts = { from_params = "theta.toml" } # every chain at the values in the file
```

The rules fall into two kinds, and the kind is what R̂ needs to know.

**Spread** rules give each chain its own start, so the between-chain R̂ can say
whether the chains found the same posterior:

| `starts`                          | where the chains start                                                                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `from_prior`                      | one draw from the declared priors per chain; base point unused                                                                                    |
| `uniform_unconstrained`           | Stan-style boundary-avoiding draws on the unconstrained scale; base point unused                                                                  |
| `lhs`                             | Latin-hypercube stratified over `bounds`; base point unused                                                                                       |
| `uniform`                         | chain 1 at the base point; the rest uniform within `bounds`                                                                                       |
| `{ from_posterior = "<handle>" }` | one row of a stored fit's posterior per chain (or of a draws TSV named directly); a parameter absent from the source falls back to bounds-uniform |

**Point** rules put every chain at one point. R̂ is then _not assessed_ — chains
that began together agree by construction, and the summary says so instead of
reporting a pass:

| `starts`                     | where the chains start                                                                                        |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `single`                     | every chain at the base point                                                                                 |
| `{ from_mle = "<handle>" }`  | every chain at a stored fit's point estimate; a parameter absent from the source falls back to bounds-uniform |
| `{ from_params = "<toml>" }` | every chain at the values in a flat params TOML                                                               |

A **handle** is any fit reference: `@label`, a fit-id prefix, a method leaf's
`run_id` prefix, the leaf directory, or a `fit.toml`. A source whose stored
verdict is not converged is refused, since starting from an unconverged fit
launders its multi-modality into this one; `fit run --allow-nonconverged-source`
records the choice to start from it anyway. A `@label` names the fit segment,
which two method files of one problem share: while it holds one leaf the label
is enough, and once it holds several the refusal lists them — name the leaf by
its directory or its `run_id` prefix.

**The default.** When the file does not say, `starts` is `from_prior` if every
estimated parameter has a prior the chains can be drawn from, and
`uniform_unconstrained` otherwise; `fit run` prints which, and why. A
bounds-uniform draw at province scale is routinely a start the filter cannot
score, because a fixed relative error in a rate is a standardised residual that
grows with the square root of the population; a prior carries the scale.

**`[estimate].start` sets the base point** — one θ. It is load-bearing under
`single` and `uniform` (and for a parameter a `from_mle` / `from_params` source
does not name); the other rules ignore it on purpose, and `fit run` notes when a
declared `start` was unused. The three spreading rules — `uniform`, `lhs`,
`uniform_unconstrained` — fall back to the base point at `chains = 1`, since
with one chain there is nothing to spread. The source-reading rules do not: they
read from their source however many chains you asked for.

`starts` is part of the method's identity: the same method under a different
rule is a different run, and a sourced rule folds the source file's content into
the run's dependencies, so a regenerated upstream re-keys the fits that started
from it. `fit run --starts <spec>` overrides the file with the same spellings
(`--starts from_prior`, `--starts from_mle=@mle`).

Every multi-chain sampler (`if2`, `pgas`, `pmmh`, `mh`, `nuts`) writes a
`chain_starts.tsv` recording where every chain actually began, before any
perturbation, and names the rule that supplied those values. That file, not the
config, is the authority on what a run did — a fit started with `from_mle`
records `from_mle` on every row, because every chain starts at the upstream
fit's single point. The optimizer-only methods (`nl-sbplx`, `nl-bobyqa`) write
none.

A spread rule is a lottery over one draw, so a start the filter cannot score is
one unlucky draw rather than a verdict on the chain: it is redrawn under the
same rule, up to ten times per chain, and the chain is refused only when every
attempt fails. Every attempt is in `chain_starts.tsv` — one row per attempt,
with `attempt`, `status` (`accepted`, `rejected`, or `refused`), the filter's
`ess` at refusal and the `reason` — and the header line counts the redraws
(`retried=K`). A point rule has nothing to redraw and gets its one try.

> **Unknown keys are rejected.** A misplaced or misspelled key is a hard error
> naming the offending key — `fit.toml` is parsed strictly. A top-level `dt` (it
> belongs in `[config]`) or `particle` (it is `particles`, under `[method]`)
> fails at load rather than being silently dropped, so a sweep that varies a
> typo'd knob can't quietly produce identical fits.

## Migrating a `[stages]` file

A file that still carries the stage map is refused at load, and the message
spells the rewrite for that file:

```
legacy table `[stages.posterior]`
  replacement: rename to `[method]` and run it with
    camdl fit run fit.toml
  a file carries one `[method]`; put `[stages.scout]` in its own file
  `init_mle = "scout"` has no replacement in the file: it started every chain at
  scout's point estimate, which makes R̂ uninformative. Run scout first and, if a
  warm start is wanted, write one of
    starts = { from_posterior = "@scout" }   # one draw per chain (keeps R̂ meaningful)
    starts = { from_mle = "@scout" }         # every chain at one point (R̂ not assessed)
  `init = "lhs"` becomes `starts = "lhs"`
  See `camdl docs fit-toml`.
```

Nothing is converted silently. `init` is `starts`; `init_mle` is one of the two
sourced rules, chosen by the author, with the upstream run first and named by
handle; `survey_top_k` with `survey_path` / `survey_top_k_n` is gone — a survey
landscape is not a posterior — and `from_prior`, or `from_posterior` from a
short run, takes its place; the top-level `fit_starts` key, which no runner
read, is gone with it (`from_prior` is now the default whenever every estimated
parameter declares a prior).

## Where scoring begins

An **incidence** observation (a weekly case count, say) is the flow accumulated
over the period its row covers, and the model states that period on the stream —
`covers = day(time)`, `starting_on(...)`, `ending_on(...)`, `closing_at(...)`,
or `window_start`/`window_stop` columns (`camdl docs data`, "What a row
covers"). That declaration also decides where the likelihood begins. The first
row's period opens where the declaration says, so a model whose dynamics start
well before the data — `simulate.from` in 2011 so births and SIA/MCV covariates
shape the susceptible pool, case data from 2014 — simulates the 2011–2014 span
with the full stochastic dynamics and scores none of it. The first datum is
scored against its own declared period, never against the whole warm-up, and
there is no `fit.toml` setting for that boundary. A first period that opens
before `simulate.from` is an error naming both times: that span is never
simulated.

`fit.toml` used to carry a `condition_from` key for this. It is removed, and a
leftover key — top-level, or under `[data]` — is a hard error rather than an
unknown-field rejection:

```
condition_from = ... is no longer a fit.toml key: an incidence stream now states
what each row covers (`covers = ...` or window columns in the model), and a
declared first period opens where it says, so the warm-up before the first row
is discarded without a separate setting. Delete the key.
```

`camdl pfilter`, `camdl profile` and `camdl fit predict` bind the same declared
periods, so a fixed-θ log-likelihood or a predictive row covers exactly the
window the fit scored. The history is in `camdl docs language-changes`.

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
declaration; if neither is present, a Bayesian method falls back to flat **with
a warning** — camdl refuses _silent_ implicit-flat priors, because the prior
shows up in the posterior. The explicit `{ flat = {} }` is how you say "flat
here, on purpose" without the warning. A flat prior is not a distribution the
chains can be drawn from, so a parameter left flat also moves the default
`starts` from `from_prior` to `uniform_unconstrained`.

## Method algorithms

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
[method]
algorithm = "nuts" # or "mh" for the gradient-free sampler (`iterations` + `burn_in`)
backend = "ode"
chains = 4
warmup = 500 # step-size adaptation draws (discarded)
samples = 500 # posterior draws kept per chain
```

Common to every method: `starts` (above), and the `dt_check` sub-table where the
method runs a dt-convergence audit.

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
`[fixed]`, and a `[method]`.
