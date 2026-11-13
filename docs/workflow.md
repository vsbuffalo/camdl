# The fit workflow

This page is the order of work for taking a model to a fit whose results you can
report: check the dynamics, show that the pipeline recovers known values from
synthetic data, choose a method, fit, read the diagnostics, and compare the
fitted model with the data. It is written for modelers and for the AI agents
that run camdl for them. Notes meant only for agents are in slate boxes, which
the website collapses and `camdl docs workflow` prints in full.

The reasoning behind the steps, including identifiability and what priors are
for, is in [`camdl docs concepts`](concepts.md). How each algorithm works is in
[`camdl docs inference`](inference.md), and every `fit.toml` field is in
[`camdl docs fit-toml`](fit-toml.md).

## At a glance

<!-- figure: fit-workflow -->

```
write model → check → simulate → synthetic recovery → choose a method
            → write fit.toml → fit run → fit summary → fit predict → validate

optional, when a fit needs them:
  survey (map the likelihood surface) · if2 (a point estimate, profiles)
```

<!-- /figure -->

## 1. Compile and sanity-check the dynamics

```bash
camdl check model.camdl
camdl simulate model.camdl --params p.toml --obs sim.tsv --seed 1
```

`simulate` defaults to the `chain_binomial` backend, the same default `fit`
uses, so a forward simulation at a fitted point reproduces the fit's dynamics.
Pass `--backend gillespie` (exact stochastic simulation) or `--backend ode`
(deterministic) to switch. The `observations {}` block that scores real data in
a fit also samples synthetic data here through `--obs`, which is why the next
step is a valid test. Look at `sim.tsv`: is the epidemic's timing, peak and
final size plausible?

## 2. Recover known values from synthetic data

Simulate data from parameter values you choose, fit the model to it, and check
that the fit recovers those values. Do this before fitting real data: it
separates "my pipeline works" from "my model describes the world". Passing it is
necessary, not sufficient.

```bash
camdl simulate model.camdl --params truth.toml --backend chain_binomial \
    --dt 0.5 --seed 7 --obs synth.tsv
camdl fit run fit_synth.toml --label synth --seed 0
camdl fit summary @synth
```

If the posterior does not cover the true values, stop: the problem is in the
pipeline (the model's structure, the step size `dt`, or the observation model),
not in the data. Choose `dt` small relative to the fastest rate, a few steps per
mean dwell time; when unsure, halve it and compare. A coarse `dt` biases
estimates in a way that recovery at the same `dt` will not reveal (the reason is
in [`camdl docs concepts`](concepts.md)).

## 3. Choose a method

For a stochastic model, fit a posterior with PGAS, and use PMMH when PGAS
struggles or as a check on it. Use IF2 and the optimizers for what a point
estimate is good for, not as the final fit, because a point estimate hides the
identifiability problems a posterior shows.

| You want                                      | `algorithm`                           | Backend          | It returns                                 |
| --------------------------------------------- | ------------------------------------- | ---------------- | ------------------------------------------ |
| The posterior of a stochastic model (default) | `pgas`                                | `chain_binomial` | Parameter and latent-trajectory draws      |
| A second posterior sampler, or PGAS struggles | `pmmh`                                | `chain_binomial` | Parameter draws                            |
| A maximum-likelihood estimate, or profiles    | `if2`                                 | `chain_binomial` | One point per chain, and chain agreement Â |
| The posterior of a deterministic (ODE) model  | `mh`                                  | `ode`            | Parameter draws                            |
| A maximum-likelihood estimate of an ODE model | `nl-sbplx`, `nl-bobyqa` or `nl-lbfgs` | `ode`            | One point per start                        |
| The log-likelihood at fixed parameter values  | `pfilter`                             | `chain_binomial` | Log-likelihood estimate and filter health  |

The ODE methods fit a different likelihood from the stochastic ones: the
likelihood of the data around one deterministic trajectory, which is appropriate
when the population is large enough that chance events do not shape the epidemic
([`camdl docs inference`](inference.md), "Fitting on the ODE backend").

**A posterior shows what the data cannot tell apart.** A point estimate says
where the likelihood is highest, not how sharply it peaks. When two parameters
trade off along a ridge, such as a seed size and an introduction time, or a
transmission rate and a reporting rate, an optimizer still returns a single
point, and nothing in that point says the data could not separate them. A
posterior sampler draws the whole ridge: the pair plot shows the correlation, a
marginal posterior that looks like its prior shows a parameter the data did not
inform, and R̂ fails when the chains spread along the ridge (section 6 works
through a real case). The same information can be had from IF2 through profile
likelihoods (`camdl profile`), but one parameter or one pair at a time, with a
separate maximization at every grid point (Ionides et al. 2017, _J. R. Soc.
Interface_ 14: 20170126). A posterior gives every pair in one run.

**PGAS and PMMH differ in how they handle the latent epidemic.** PGAS (particle
Gibbs with ancestor sampling) holds one sampled trajectory of the epidemic
fixed, scores the parameters against it exactly, and moves them with NUTS using
gradients the compiler derives, then redraws the trajectory. It returns
posterior epidemic trajectories as well as parameters. Its weak point is a long
series, where a new trajectory can differ from the old one at only a few steps
and the chain moves slowly; trajectory renewal, in the diagnostics, measures
this. PMMH (particle marginal Metropolis–Hastings) averages over trajectories
with a particle filter and accepts or rejects whole parameter proposals on the
filter's log-likelihood estimate. That estimate is noisy, so PMMH needs more
particles as the series grows and mixes more slowly, but it shares no machinery
with PGAS beyond the model. When the two agree on a posterior, that is evidence
neither is stuck. Both require a prior on every estimated parameter.

**IF2 is still useful, for a narrower job.** Use it when you want a likelihood
estimate without priors, to profile a parameter you suspect is poorly
identified, or as a quick first look at whether the data constrain the model at
all. Do not use it to place the chains of the final posterior:
`starts = { from_mle = "@mle" }` starts every chain at one point, and R̂, which
needs chains that start apart, is then reported as not assessed.

::: {.agents title="Ask before choosing a prior shape"}

A posterior sampler needs a prior on every estimated parameter, and the prior
shows up in the posterior. If the model's author has not documented one, ask
rather than choosing a shape to make the sampler run. See
[`camdl docs agents`](agents.md), "When to stop and ask the human".

:::

### Optional: map the likelihood surface with `survey`

```bash
camdl survey model.camdl --fit fit.toml --render
```

`survey` evaluates the likelihood at Latin-hypercube points across the parameter
bounds and writes them to `landscape.tsv`, with `--render` adding a pair plot of
the points in `landscape.html`. It shows separate high-likelihood regions,
ridges, and parameters whose best values sit at a bound. It is worth running
when:

- the model estimates many parameters, so starts drawn from the priors may land
  far from where the likelihood is high;
- a fit's chains disagree, or one chain's log-likelihood differs from the
  others', and you want to know whether there is more than one high-likelihood
  region;
- the priors are weak and you need to know where the likelihood puts the mass.

For a small model with informative priors whose chains already agree, it adds
little.

::: {.agents title="Surface the survey plot rather than reading its geometry"}

Rank points by the log-likelihoods in `landscape.tsv`, not by your reading of
the scatter in `landscape.html`. Give the person the HTML path and your one-line
reading, and let them confirm before you start a fit from it.

:::

## 4. Write `fit.toml` and run the fit

A `fit.toml` names the model, the data, what to estimate, what to fix, and the
one method that fits it. A minimal, complete one:

```toml
[model]
camdl = "model.camdl"

[data.observations]
cases = "data/cases.tsv" # one key per observation stream in the model

[estimate] # bounds always; a posterior sampler needs a prior (here, or in the model via ~)
beta = { bounds = [0.001, 0.5], start = 0.04, prior = { log_normal = { mu = -2.0, sigma = 1.0 } } }
gamma = { bounds = [0.01, 1.0], start = 0.12, prior = { log_normal = { mu = -1.2, sigma = 0.5 } } }

[fixed]
rho = 0.6
k = 10.0

[method] # one per file; `algorithm` picks the method
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 600
sweeps = 300
# starts = "from_prior"   # the default here: every parameter has a prior
```

- **One `[method]` per file.** `algorithm` picks the method from the table in
  section 3. Everything above `[method]` is the problem, which every reader
  loads (`simulate --draws prior --fit`, `pfilter --fit`, `survey --fit`,
  `profile --fit`). A second way of fitting the same problem, such as an `if2`
  point estimate beside this posterior, is a second file with the same problem
  half: `camdl fit new --from fit.toml fit-if2.toml`, then edit `[method]`.
- **Chains start apart by default.** `starts` says where the chains begin. When
  omitted it is `from_prior` (one draw per chain from the priors) whenever every
  estimated parameter has a prior, else `uniform_unconstrained`. A start from a
  stored fit is written where it is used:
  `starts = { from_posterior = "@base" }` keeps R̂ meaningful, and
  `starts = { from_mle = "@mle" }` puts every chain at one point, so R̂ is
  reported as not assessed.
- **Priors** for a `pgas`, `pmmh` or `mh` method must be explicit, declared in
  `[estimate].prior` or in the model (a `~` declaration); an implicit flat prior
  is refused. The shapes are `log_normal {mu,sigma}`, `normal {mean,sd}`,
  `beta {alpha,beta}`, `uniform` and `half_normal`.
- The full schema, with every method field, `starts`, transforms and holdout, is
  [`camdl docs fit-toml`](fit-toml.md).

```bash
camdl fit run fit.toml --label baseline --seed 1
```

`--label baseline` names the fit, so every later command can refer to it as
`@baseline` instead of by its run directory.

## 5. Read the diagnostics

```bash
camdl fit summary @baseline      # or the run directory `fit run` printed, or a hash prefix
```

Every fit command (`fit summary`, `fit predict`, `compare`) takes the same
**handle**: an `@label`, a fit hash prefix, the run directory `camdl fit run`
prints, or the `fit.toml` itself (resolved to its one run; an ambiguous match is
listed rather than guessed). Each run archives its own compiled model, so a
handle keeps working even if the `.camdl` file moves. `camdl list` enumerates
every run.

### A posterior fit

For PGAS, PMMH and `mh`, the summary leads with a verdict and the reason for it,
then prints these blocks:

- **posterior** — each parameter's mean, bulk and tail effective sample size
  (ESS), and R̂, sorted worst R̂ first so the problems are at the top.
- **convergence** — the two halves of R̂ (bulk and folded, explained below), the
  classic Gelman–Rubin statistic for comparison with older fits, the per-chain
  ESS, and the sampler's acceptance rate and ESS per iteration and per second.
- **chains** — each chain's mean log-likelihood and whether it differs from the
  others' by more than Monte Carlo error explains. A chain that differs may be
  sampling a side mode or a weakly identified ridge, or may not have mixed.

`camdl fit summary --explain` adds a short definition of every column under the
block it belongs to.

### Which R̂, and which ESS

The `rhat` camdl reports for a posterior is the rank-normalized split-R̂ of
Vehtari et al. (2021), taken as the larger of the rank-normalized split
statistic and its folded counterpart. Splitting each chain in half catches a
chain that drifts across its own run; folding (`|x − median(x)|`) catches chains
that agree on location and disagree on spread. Both are invisible to the classic
Gelman & Rubin (1992) statistic, which compares chain _means_ only, and which
each `*_summary.json` still carries as `rhat_classic` so an old fit and a new
one stay comparable.

Both halves of that `max` are reported too, as `rhat_bulk` (location) and
`rhat_folded` (spread), and which one is larger is the answer to _why_ R̂ is
high. A large `rhat_bulk` says the chains disagree about where the posterior
sits: lengthen warm-up, or discard more of it. A large `rhat_folded` says they
agree on location and disagree on how wide the posterior is, which for a
particle method points at per-chain effective particle diversity. Above the
threshold, `camdl fit summary` and the end-of-run block print the split:

```
max R̂ = 1.313  ~  NOT converged  (rank-normalized split R̂, threshold 1.05)
  beta — R̂ = max(bulk 0.998, folded 1.313); the folded half is larger —
         the chains agree on location and disagree on spread
```

No threshold is placed on the gap between the halves; it is reported, not
linted. See `docs/dev/proposals/2026-08-22-reporting-two-rhat-estimators.md` for
the evidence and for what a cutoff would need before it could be picked.

`ESS` is the rank-normalized **bulk** effective sample size, with the **tail**
ESS (the smaller of the 5% and 95% quantile-indicator ESS) beside it. Neither is
suppressed when chains disagree: both use the between-chain variance rather than
summing per-chain estimates, so they stay meaningful, and small, exactly when a
fit has not mixed. Report ESS/N with the ESS: bulk-ESS 11 out of 11,200 draws
means the estimator summed autocorrelations out to nearly the whole run and is
reporting mostly about its own truncation point.

> The healthy band below (`< 1.05`) was calibrated against the classic
> statistic. Vehtari et al. recommend `< 1.01` for the rank-normalized one, and
> `ESS > 400` before R̂ is trustworthy at all. Which band camdl certifies against
> is an open decision; see gh#84. The estimator changed; the published band has
> not.

### An IF2 fit

An IF2 summary reports a point estimate and whether the independent optimizer
chains reached the same one:

- **best loglik (loglik-eval)** — the maximum-likelihood estimate re-scored at a
  high particle count. Use this number; IF2's running log-likelihood during
  optimization is biased by the parameter perturbations.
- **The IF2 convergence gate**, two legs, both of which must pass:
  - **Â (chain agreement):** `< 1.05` passes, `1.05–1.10` is marginal, `≥ 1.10`
    fails. Did the independent optimizer chains climb to the same place?
  - **Δ_dB (decibans spread):** the spread between the best and worst chain's
    log-likelihood, against a threshold of about 30 dB that accounts for its
    standard error. Was the place they agreed on any good? Chains can pass Â
    while sitting in regions thousands of dB apart; this leg catches that.
- **per-chain loglik-eval** — each chain's re-scored log-likelihood ± its
  standard error; `← selected` marks the estimate.
- **ESS at θ̂** — the particle filter's effective sample size (`min` and `mean`
  over observation steps) at the estimate. A single-digit `min` at the most
  informative observation is tolerable; `min ≈ 1` everywhere means the
  log-likelihood at the estimate is unreliable, so raise the particle count.
- **dt-convergence (Richardson)** — the log-likelihood at the estimate re-scored
  at `dt`, `dt/2` and `dt/4`. `PASS` means the estimate survived a finer
  discretization.

**Â is not R̂.** Â asks whether optimizers converged to the same point. R̂ asks
whether posterior chains are sampling the same distribution, and camdl reports
it, by that name, only for posterior fits. Read posterior mixing from R̂, NUTS
divergences and trajectory renewal, never from Â, and do not carry R̂'s "above
1.01, keep sampling" reflex over to Â.

### Diagnostics reference

| Method           | Diagnostic           | Healthy         | Warning           | Action                                         |
| ---------------- | -------------------- | --------------- | ----------------- | ---------------------------------------------- |
| Particle filter  | ESS per-obs          | > 50% of N      | 10–50%            | more particles or looser obs model             |
| Particle filter  | ESS at MLE           | mean > 50%      | mean < 30%        | estimate `σ²` or `k`                           |
| Particle filter  | log-likelihood       | finite          | `−∞`              | check starts / model structure                 |
| IF2 (MLE)        | Â                    | < 1.05          | 1.1–1.5           | more iterations                                |
| IF2 (MLE)        | Â                    | —               | > 1.5 + LL spread | multimodal surface, more chains                |
| IF2 (MLE)        | logit position \|z\| | < 2             | > 3               | widen bounds or change transform               |
| PGAS (posterior) | R̂ (rank-normalized)  | < 1.05          | > 1.1             | more sweeps; check multimodality with `survey` |
| PGAS (posterior) | trajectory renewal   | > 30%           | < 10%             | more CSMC particles                            |
| PGAS (posterior) | NUTS divergences     | 0               | any               | reduce step size, reparameterize               |
| PGAS (posterior) | NUTS max tree depth  | < 20% of sweeps | > 50%             | increase `max_treedepth`                       |

::: {.agents title="Report a fired diagnostic; do not decide it is fine"}

When `camdl fit summary` reports a diagnostic, tell the person which one fired
and what the table says it means, rather than deciding that it is fine. Never
lower a threshold to make a gate pass.

:::

## 6. When it doesn't converge: a failing gate is information

Don't tune a failing diagnostic away. The canonical illustration is the WA-State
cryptic-introduction fit (estimating the early-COVID introduction time τ): under
uninformative priors the `n_seed` posterior pins to its upper bound and
`fit summary` reports `max R̂ = 1.216 ✗`, because the data alone can't separate a
small, late introduction from a large, early one (the `(τ, n_seed)` ridge). The
fix isn't more sweeps; it's **weakly informative structural priors** (a
`log_normal` on seed size, a `beta` on reporting) that regularize the
non-identified direction. After that, R̂ passes, the posterior is unimodal, and τ
agrees with the independent Bedford genomic estimate. The failing R̂ was
_correct_: it said "the data don't identify this", not "run longer". An IF2 fit
of the same model would have returned one point on the ridge.

The full case study, with prior-predictive checks, the ridge, PSIS-LOO,
prequential elpd, and mechanism and prior sensitivity, is in
[`camdl docs concepts`](concepts.md).

## 7. Validate the fit

### Predicted-vs-observed in one verb: `camdl fit predict`

The workhorse posterior check — overlay the observed data on the fitted model's
predictive band — is a single verb that reads the fit and writes a tidy,
plot-ready artifact:

```bash
camdl fit predict @baseline --stream onset
# wrote results/fits/sle-8a3f12b4/predictive/onset.tsv
# wrote results/fits/sle-8a3f12b4/observed/onset.tsv
```

`fit predict` resolves the fit's canonical post-warm-up posterior draws,
forward-simulates each draw through the **real** observation model (sampling
`y_rep`, not the projected mean, so observation noise is in the band),
integrates over the cloud by pooling and quantiling, and writes two files under
the run directory:

- `predictive/<stream>.tsv` —
  `time | <dims…> | horizon | treatment | fit_rhat_max | fit_ess_min | rhat_mean | ess_mean | rhat_pred | ess_pred | n_draws | q05 q25 q50 q75 q95`.
  The `q05…q95` columns are the ribbon; the `horizon` and `treatment` columns
  make the two predictive axes explicit (so an honestly-wide posterior band is
  never confused with a narrow plug-in one), and `fit_rhat_max`/`fit_ess_min`
  carry the fit's own convergence numbers alongside every band. **`fit_rhat_max`
  is the rank-normalized split R̂ and `fit_ess_min` the bulk-ESS** (see
  [Which R̂, and which ESS](#which-r-and-which-ess)); `fit_ess_min` is left empty
  when any assessed parameter has no pooled ESS, rather than minimizing over the
  ones that do. The sibling `predictive.json` tags this contract
  `camdl.predictive/v3`. Earlier tags are not join-compatible with it and **must
  not be joined without keying on the tag**: `/v2` spelled these two columns
  `rhat_max`/`ess_min` and had no per-row channels, and `/v1` carried classic
  Gelman–Rubin R̂ and a per-chain Geyer sum under those same two names — the same
  header, a different statistic.

  `fit_rhat_max`/`fit_ess_min` describe **the fit**, not the row — which is what
  their names say. They are the worst parameter's numbers, repeated identically
  down the file. The two pairs beside them describe **the row**, and are what a
  decision to publish a curve or a forecast should rest on:

  | column pair              | reduces                                    | answers                                                     |
  | ------------------------ | ------------------------------------------ | ----------------------------------------------------------- |
  | `rhat_mean` / `ess_mean` | the latent expected value `E[y \| x_t, θ]` | do the chains agree about the **expected trajectory** here? |
  | `rhat_pred` / `ess_pred` | the predictive draws `y_rep`               | do the chains give the same **predictive distribution**?    |

  **Decide on `rhat_mean`.** A predictive draw carries observation noise, and
  that noise lands in the within-chain variance; where it is comparable to the
  between-chain disagreement it swamps the numerator and `rhat_pred` is pulled
  toward 1 however much the chains disagree. Chains whose eight-week forecasts
  span 93 to 372 cases per day can still show `rhat_pred` near 1. The dilution
  grows with the observation dispersion, so it is worst exactly where
  mechanistic models live. Read `rhat_pred` only when the interval you are
  quoting is genuinely dominated by irreducible observation noise. An empty cell
  is a refusal — fewer than 2 chains, fewer than 4 draws per chain, a
  `draws.tsv` with no chain column, or a row that never moved — never a pass;
  the `one_step` horizon leaves both pairs empty, because its cell pools over
  filter particles as well as draws.

  `--by-chain` is the follow-up once `rhat_mean` has flagged a row: it adds a
  leading `chain` column and one extra band per chain beside the pooled `all`
  rows, on **both** horizons, so you can see _which_ way the chains disagree.
  Overlapping per-chain forward bands mean the pooled band summarises one
  forecast; separated ones mean it is a mixture of several, and quoting its
  quantiles reads as uncertainty where the truth is disagreement — opposite
  actions. The `one_step` per-chain bands ask the in-sample version of the same
  question — does each chain explain the record you already have? — and, being
  re-anchored to the data at every step, they separate disagreement about the
  fitted trajectory from extrapolation uncertainty, which the free-forward bands
  cannot. A per-chain row carries no `rhat_*`/`ess_*` cell (those compare
  chains) and reports its own `n_draws`. Without the flag no `chain` column is
  written.

  `quantities/<name>.tsv` carries the same reduction under `rhat` / `ess` — one
  pair, since a quantity has a single value per draw. Those are the reported
  estimands, so they are the first numbers to read. A quantity over latent state
  or derived arithmetic is noise-free and its `rhat` is the undiluted kind; a
  quantity whose manifest `source` is `observations` reduces sampled `y_rep`,
  carries observation noise, and so reads like `rhat_pred`. `simulate --draws`
  has no chains behind it and writes neither column.

  The two per-row pairs and the parameter R̂ can disagree in either direction,
  and both directions are ordinary. A reportable quantity is often far better
  determined than the parameters behind it, so a fit with `fit_rhat_max` near
  2.7 can carry a forecast whose `rhat_mean` sits near 1.05.
- `observed/<stream>.tsv` — `time | <dims…> | value`, the recorded series in the
  same tidy keys.

A consumer reads both files, joins on `(time, <dims>)`, and plots `observed`
over the `predictive` ribbon — one facet per stratum, with no run-store, DSL, or
likelihood knowledge:

```python
import polars as pl
pred = pl.read_csv(".../predictive/onset.tsv", separator="\t")
obs  = pl.read_csv(".../observed/onset.tsv",   separator="\t")
```

**Two horizons answer two questions** (see
[`camdl docs diagnosing-fits`](diagnosing-fits.md) §5). Omit `--horizon` to emit
all applicable for the fit's backend:

- `--horizon free_forward` — the generative check: replay the fitted model from
  the start, never re-anchored to data. The harshest test, and what exposes
  generative misspecification. Available on any backend.
- `--horizon one_step` — the honest short-horizon forecast `p(y_t | y_{1:t-1})`:
  re-condition on the data each step. Chain-binomial only (an ODE fit's one-step
  is identical to its free-forward and is refused with a redirect). Pools over a
  posterior subsample (`--n-draws`, default 200); both horizons stack in the
  same file, distinguished by the `horizon` column.

`fit predict` refuses an optimizer fit (IF2 / NLopt) with an actionable message
— such a fit returns one best-fit point, not a distribution, so there is no
posterior band to draw; get its parameters with
`camdl fit summary <run>
--params-only` and run a plug-in
`camdl simulate --params …` instead.

If the model declares a `quantities {}` block, `fit predict` also bands each
derived quantity over the same posterior draws into `quantities/<name>.tsv` with
a `quantities.json` manifest. These are either **series** — a channel the model
computes but doesn't track as a compartment (force of infection, effective
reproduction number, cumulative incidence, EIR), one value per output time — or
**scalar** summaries (peak size, attack rate, time-to-peak): the posterior of a
reported quantity, not just of the fitted series. Declare them in the model
rather than reconstructing them in a downstream script; see
[`camdl docs user-features`](user-features.md) ("Reporting derived quantities").

### Other validation steps

```bash
# Prior-predictive — do the priors imply plausible epidemics?
camdl simulate model.camdl --draws prior --fit fit.toml -n 200 --obs prior_ppc.tsv

# Identifiability — profile a suspect parameter (1D or 2D)
camdl profile model.camdl --particles 1500 \
    --fixed gamma=0.1 --sweep "tau=lin(-60,5,12)" --fit fit.toml

# Model comparison — prequential predictive scoring, straight from two fits
camdl compare @baseline @candidate --particles 2000 --seed 7
```

`camdl compare` ranks models by prequential elpd / CRPS / PIT. Passed two fit
handles, it **auto-derives** each model's prequential at θ̂ via `pfilter` — the
same particle count and seed for both, so the scores stay commensurable — so you
no longer hand-run `pfilter --save-prequential` first. (You still can: an
explicit `prequential.json` path is read as-is, for a custom filter
configuration.) Its scores are **plug-in and in-sample-optimistic** — computed
at a single θ that was fit to the whole series — so they are not a
leave-future-out forecast score. Nor does the optimism cancel in a difference:
it grows with the effective number of parameters a model was free to tune
against the same observations, so Δelpd tilts toward the more flexible model.
Treat a comparison as indicative rather than decisive, and read it with `se(Δ)`
— when `|Δelpd| < 2·se(Δ)` the evidence column says `within noise` and gives no
tier. `compare` prints these caveats with the table on every run.

## Decisions that belong to the modeler

Running the steps above is mechanical. Four decisions along the way are not, and
they belong to the person responsible for the model:

- **Choosing a prior shape** for a parameter. A `normal(0, 1)` picked to make a
  sampler run goes straight into the posterior.
- **Responding to a failed convergence gate.** Find out why it failed (the
  bounds, too few chains or iterations, more than one mode) rather than lowering
  the threshold.
- **Reading the geometry of a plot**: the `survey` pair plot, a posterior pair
  plot, a predictive ribbon.
- **An external estimate that disagrees**, such as a genomic estimate or a
  published result. Raise the disagreement rather than tuning until the numbers
  match.

::: {.agents title="Stop and ask at each of these decisions"}

Pause at each decision above and put it to the person, with the evidence you
have, rather than deciding it yourself. The full list of when to pause, when to
flag and proceed, and what is safe to do alone is in
[`camdl docs agents`](agents.md), "When to stop and ask the human".

:::
