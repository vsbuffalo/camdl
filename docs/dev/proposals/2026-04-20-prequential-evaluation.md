---
status: proposal
date: 2026-04-20
authors: camdl-book authors
---

# Prequential Out-of-Sample Evaluation in camdl

**Methods documentation and user-facing design**

_Status: draft. Target release: post-CR18. Scope: SSM/POMP inference pipelines
(IF2/MLE, PMMH, CPM-PMMH, PGAS/NUTS). Audience: camdl contributors and users
running model comparisons for publication._

---

## Release plan: two stages

This proposal is split into two stages to keep the first-release scope
tractable. **Part I** is the minimum viable prequential pipeline that unblocks
the boarding-school chapter and supports plug-in MLE workflows (IF2 + profile
likelihood). **Part II** is the Bayesian and panel-data extension, deferred to
a later release.

| Feature | Stage |
|---|---|
| `PrequentialTrace` as first-class output of `camdl fit` | Part I |
| Plug-in predictive from bootstrap PF (IF2/MLE) | Part I |
| Log score + CRPS | Part I |
| PIT coverage at 50% / 90% intervals | Part I |
| Effective sample size as diagnostic | Part I |
| Testing-by-betting e-process (as display mode on `compare`) | **Part I** |
| `camdl compare` with baseline-centered Δelpd + ΔCRPS table | Part I |
| Structural-fairness preflight (data hash, obs model, etc.) | Part I |
| Anti-pattern detection (t₀=0, seed collisions, etc.) | Part I |
| Named fits + hash-prefix resolver | Part I |
| JSON output schema as plotting contract | Part I |
| Fully Bayesian predictive (LFO-PSIS) | Part II |
| Pseudo-posterior provenance from IF2 cloud | Part II |
| PSIS $\hat k$ diagnostic | Part II |
| Randomized PIT for discrete observations | Part II |
| Identifiability sweep for $t_0$ (opt-in then default) | Part II |
| Rolling-origin $k$-step-ahead | Part II |
| Energy score / panel comparison UX | Part II |

Decisions on prioritization: e-values are in Part I because the computation is
free (a cumulative sum on quantities the PF already emits) and the pedagogical
framing — "stochastic model's predictives took deterministic model's $1 to
$59" — belongs in the first user-facing release of prequential comparison.
LFO is in Part II because all current camdl inference pipelines that matter
for Part I's worked example (IF2/MLE, profile likelihood) are plug-in; LFO is
only needed when PMMH/PGAS become default.

---

# Part I — v1 scope (this release)

## 1. Motivation

The standard small-sample time-series comparison workflow — single train/test
split, point-estimate fit, RMSE on held-out points — has two failure modes that
matter for compartmental epidemic models:

1. **Split-noise dominance.** With $n \approx 14$ (boarding school), the scalar
   holdout score is a random variable with variance comparable to the
   differences between competing models. A single split cannot detect this and
   does not report it.
2. **Split-location confounding.** Placing the peak and post-peak dynamics in
   training (a natural choice when the goal is "fit the outbreak") means the
   decline-limb holdout is tested against parameters already pinned down by what
   it is supposed to evaluate. This is closer to smoothing than forecasting.

Prequential evaluation (Dawid 1984) solves both. For state-space models fit with
SMC, it is essentially free: the bootstrap particle filter already computes the
one-step predictive distribution at every assimilation step. This document
specifies how camdl exposes those quantities as first-class output, the scoring
rules applied to them, the diagnostics produced alongside, and the command-line
UX for comparing fits.

The design goal is that a user running `camdl fit` receives a `PrequentialTrace`
automatically, and `camdl compare <refs...>` produces a publication-ready table
without rerunning any inference.

---

## 2. Prequential principle

For observations $y_{1:T}$, the joint log-likelihood factorizes into a sequence
of one-step forecasts:

$$
\log p(y_{1:T}) = \sum_{t=0}^{T-1} \log p(y_{t+1} \mid y_{1:t}).
$$

Each term is a prediction made using only the past. Dawid's (1984) insight:
calibration of a probabilistic forecaster can be assessed from this sequence
alone, without repeated sampling of whole trajectories. For single-outbreak data
— one realization, moderate $T$ — this converts a dataset that looks like "one
data point" into $T - t_0$ forecast–outcome pairs.

camdl adopts the prequential decomposition as the default out-of-sample
procedure for all SSM pipelines. Alternative procedures (single split,
rolling-origin $k$-step-ahead, hv-block CV) are supported but opt-in.

### 2.1 The role of $t_0$

The prequential sum starts at an index $t_0$, not at $t=0$:

$$
\text{elpd}_{\text{preq}} = \sum_{t=t_0}^{T-1} \log \hat p(y_{t+1}\mid y_{1:t}).
$$

Observations $y_{0}, \ldots, y_{t_0}$ are fed to the particle filter to
initialize the filtering distribution but are **not scored**. Scoring begins at
$y_{t_0+1}$, with the first genuine one-step-ahead forecast.

This is necessary for two reasons:

1. **Initialization noise.** The very first "forecast"
   $\log\hat p(y_1 \mid \emptyset)$ is made from the prior/initial-condition
   cloud, before the filter has seen any data. That score tests the prior, not
   the dynamics. The next few are dominated by initial-condition uncertainty
   rather than model quality.
2. **Identifiability.** For compartmental models, parameters like $R_0$ are
   only weakly identified until the filter has seen enough rising-limb
   observations. Prequential scores made against an unidentified MLE inherit
   the non-identifiability as score variance that swamps inter-model
   differences.

$t_0$ is thus the filter analogue of MCMC warmup: samples before $t_0$ are
consumed by the inference procedure and do not contribute to the scored
output. The choice of $t_0$ materially affects the score — setting it too low
makes prequential comparison unreliable — so §7.1 specifies how it is
defaulted and validated.

### 2.2 Recommended reading

- Dawid (1984), "Statistical theory: the prequential approach," _JRSS-A_ —
  foundational.
- Tashman (2000), "Out-of-sample tests of forecasting accuracy," _IJF_ — the
  standard review.
- Bergmeir, Hyndman & Koo (2018), "A note on the validity of cross-validation
  for evaluating autoregressive time series prediction," _CSDA_ — when k-fold CV
  is and isn't valid under serial dependence.
- Bürkner, Gabry & Vehtari (2020), "Approximate leave-future-out
  cross-validation for Bayesian time series models," _Statistics and Computing_
  — the computational recipe adopted here.
- Hyndman & Athanasopoulos, _Forecasting: Principles and Practice_, ch. 5 (free
  online) — rolling-origin evaluation, introductory.

---

## 3. The one-step predictive in state-space models

With latent states $x_t$, parameters $\theta$, and observations $y_t$:

$$
p(y_{t+1} \mid y_{1:t}) = \int p(y_{t+1} \mid x_{t+1})\, p(x_{t+1} \mid x_t, \theta)\, p(x_t, \theta \mid y_{1:t}) \, dx_t\, dx_{t+1}\, d\theta.
$$

Three things are being integrated: the observation model, the one-step dynamics,
and the joint state–parameter filtering posterior at time $t$. How that
integration is performed determines the quality and cost of the score.

### 3.1 Plug-in predictive (IF2, MLE)

For a point estimate $\hat\theta$, the predictive collapses to the bootstrap
particle filter's incremental likelihood:

$$
\hat p(y_{t+1} \mid y_{1:t}, \hat\theta) = \sum_{i=1}^N w_t^{(i)}\, p(y_{t+1} \mid x_{t+1}^{(i)}),
$$

where $x_{t+1}^{(i)} \sim p(\cdot \mid x_t^{(i)}, \hat\theta)$ is the one-step
propagation of particle $i$ and $w_t^{(i)}$ are the normalized filter weights at
time $t$. This quantity is emitted at every filter step by camdl's Rust SMC
backend at zero additional cost — it is a standard diagnostic of the bootstrap
filter.

Plug-in predictives are proper only under the assumption that $\hat\theta$
captures all parameter uncertainty. At $n \approx 14$ this assumption is
violated. Plug-in scores should be labeled as such in output.

### 3.2 Fully Bayesian predictive — deferred to Part II

Marginalizing $\theta$ over its partial-data posterior at each $t$ is the
Bayesian version of the one-step predictive. For plug-in pipelines this
section does not apply; users running IF2/MLE or profile-likelihood workflows
in v1 consume only the §3.1 plug-in form. LFO-PSIS, PMMH/PGAS specifics, and
the $\hat k$ diagnostic are spec'd in Part II (§10).

### 3.3 Why provenance matters

The `PrequentialTrace` carries a provenance flag so that downstream outputs
(plots, tables, methods text in the white paper) cannot drift out of sync
with what was actually computed. In v1 the enum has a single value,
`PlugIn`; Part II extends it with `LfoPsis`, `FullyBayesian`, `FullRefit`,
and `PseudoPosterior`. A plug-in predictive that silently gets reported as
"Bayesian predictive" is a defensible-for-training but
indefensible-for-publication result. We fail loud.

---

## 4. Scoring rules

camdl applies proper scoring rules (Gneiting & Raftery 2007) to the one-step
predictive. Two are reported by default; a third is planned for spatial panel
data.

### 4.1 Log score

$$
S_{\text{log}}(F, y) = \log f(y),
$$

where $f$ is the predictive density at the observation. The prequential log
score is

$$
\text{elpd}_{\text{preq}} = \sum_{t=t_0}^{T-1} \log \hat p(y_{t+1}\mid y_{1:t}).
$$

**Properties.** Local (depends only on the density at $y$). Equivalent to KL up
to a data-dependent constant. Gives the e-value / testing-by-betting connection
directly (see §6).

**Pathology.** Diverges to $-\infty$ under particle depletion or thin-tailed
predictives. At small $T$, a single bad day can dominate the total score. This
is not a bug — it is the log score doing its job — but it makes log score alone
a fragile tool for ranking at $T = 14$.

### 4.2 Continuous ranked probability score (CRPS)

$$
\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \big(F(z) - \mathbb{1}\{z \geq y\}\big)^2\, dz.
$$

$F$ is the predictive CDF; $\mathbb{1}\{z \geq y\}$ is the degenerate CDF at
$y$. CRPS measures squared $L^2$ distance between the forecast CDF and the ideal
"all mass on the truth" CDF. Two equivalent forms are useful:

**Energy form,** for independent $Y, Y' \sim F$:

$$
\text{CRPS}(F, y) = \mathbb{E}_F|Y - y| - \tfrac{1}{2}\mathbb{E}_F|Y - Y'|.
$$

First term: how close forecast draws land to the observation. Second: penalty
for diffuse predictives. Propriety is immediate from this form.

**Quantile form** (Laio & Tamea 2007):

$$
\text{CRPS}(F, y) = 2 \int_0^1 \text{QL}_\tau(F^{-1}(\tau), y)\, d\tau,
$$

where $\text{QL}_\tau$ is the pinball loss at quantile level $\tau$. CRPS is
thus the average pinball loss over all quantile levels — it scores the entire
predictive quantile function simultaneously.

**Sample estimator,** for particle draws
$\tilde y^{(1)}, \ldots, \tilde y^{(S)}$:

$$
\widehat{\text{CRPS}} = \frac{1}{S}\sum_s |\tilde y^{(s)} - y| - \frac{1}{2S^2}\sum_{s,s'}|\tilde y^{(s)} - \tilde y^{(s')}|.
$$

Naive cost $O(S^2)$; camdl uses the $O(S \log S)$ sorted-sample identity

$$
\widehat{\text{CRPS}} = \frac{2}{S^2}\sum_{s=1}^S (\tilde y^{(s)}_{\text{sorted}} - y)\left(S \cdot \mathbb{1}\{y < \tilde y^{(s)}_{\text{sorted}}\} - s + \tfrac{1}{2}\right).
$$

**Properties.** Non-local (uses the whole CDF). In the units of $y$ —
interpretable to non-statisticians ("model A beats B by 3.2 cases/day").
Degrades gracefully under tail thinness: nearby mass earns partial credit.
Robust at small $T$.

**Tradeoff.** No clean betting-game interpretation. Slightly rewards calibrated
medians over calibrated tails.

### 4.3 Why report both

If log score and CRPS rank two models identically, you have a robust ranking. If
they disagree, you have a _diagnostic_: typically one model is
sharp-but-miscalibrated in the tails (log score penalizes heavily, CRPS mildly)
and the disagreement is the signal, not the noise. camdl surfaces both by
default. Hiding one behind a "headline metric" would discard this information.

### 4.4 Energy score — deferred to Part II

For multivariate / panel predictives, CRPS generalizes to the energy score;
see §11. v1 handles univariate observations only. The `PrequentialTrace`
tensor shape (§7) is defined from the start to admit multivariate samples so
that energy-score support can drop in later without schema migration.

### 4.5 Recommended reading

- Gneiting & Raftery (2007), "Strictly proper scoring rules, prediction, and
  estimation," _JASA_ — the canonical reference.
- Laio & Tamea (2007), "Verification tools for probabilistic forecasts of
  continuous hydrological variables," _HESS_ — CRPS as average pinball loss.
- Bracher et al. (2021), "Evaluating epidemic forecasts in an interval format,"
  _PLoS Comp Biol_ — scoring rules applied to infectious disease specifically,
  good for the polio vignette.

---

## 5. Diagnostics

Scalar scores hide failure modes. camdl outputs diagnostics alongside every
`PrequentialTrace`.

### 5.1 Probability Integral Transform (PIT)

For continuous predictives, $u_t = F_{t+1\mid t}(y_{t+1})$ should be uniform on
$[0,1]$ under correct calibration (Dawid 1984; Gneiting, Balabdaoui & Raftery
2007). camdl stores $\{u_t\}_{t=t_0}^{T-1}$ and reports:

- A PIT histogram (for human inspection — the fastest way to spot systematic
  over- or under-dispersion).
- Coverage at the 50% and 90% central predictive intervals — single numbers that
  catch the most common failure mode (plug-in overconfidence).

For discrete observations (count data), point-estimate PIT exhibits stair-step
artifacts near integer values. v1 uses point-estimate PIT with this caveat
documented in the histogram output; the randomized PIT of Czado, Gneiting &
Held (2009) that removes these artifacts ships in Part II (§12).

### 5.2 PSIS $\hat k$ — deferred to Part II

LFO-specific; the diagnostic is only meaningful when the Bayesian predictive
pipeline of §10 is active. v1 has no $\hat k$ field on `PrequentialTrace`.

### 5.3 Effective sample size

For SMC-based predictives, the filter's ESS at each $t$ is stored. ESS collapse
is a mechanistic diagnostic that often precedes CRPS anomalies and should be
checked before statistical explanations are reached for.

---

## 6. Testing by betting: e-values

The log-score prequential sum has a direct interpretation as the log-bankroll of
a sequential bet, and this is worth teaching explicitly because it makes model
comparison concrete and anytime-valid.

### 6.1 Construction

An **e-value** for forecast $F_t$ and observation $y_t$ is a non-negative
statistic satisfying $\mathbb{E}_{F_t}[E_t] \leq 1$ under the null that $F_t$ is
well-calibrated. The canonical construction is a likelihood ratio
$E_t = g_t(y_t)/f_t(y_t)$ for an alternative predictive $g_t$ (Ramdas et al.
2023).

The **e-process** is the running product

$$
\mathcal{E}_T = \prod_{t=t_0}^{T-1} E_t = \prod_{t=t_0}^{T-1} \frac{g_t(y_t)}{f_t(y_t)},
$$

which is a nonnegative supermartingale under the null. Ville's inequality gives

$$
\Pr\left(\sup_{T} \mathcal{E}_T \geq 1/\alpha\right) \leq \alpha,
$$

an **anytime-valid** test: the analyst may stop as soon as
$\mathcal{E}_T \geq 1/\alpha$ and control Type I error at level $\alpha$, with
no multiple-testing correction required.

### 6.2 Connection to prequential log score

$$
\log \mathcal{E}_T = \sum_{t = t_0}^{T-1} [\log g_t(y_t) - \log f_t(y_t)] = \text{elpd}_{\text{preq}}^{(g)} - \text{elpd}_{\text{preq}}^{(f)}.
$$

The log-e-process is exactly the paired difference of prequential log scores.
camdl already computes this as a byproduct of the `compare` table (see §8) —
the e-value framing is a reinterpretation, not a new computation. It surfaces
as columns when the user passes `--show=betting`, not as a separate subcommand
path.

### 6.3 Why teach this

For small-sample scientific settings, the e-value framing has three pedagogical
advantages over fixed-$\alpha$ null-hypothesis testing:

1. **Concrete interpretation.** "Model $g$ would have turned $1 into
   $\mathcal{E}_T$ betting against model $f$" is a sentence a deputy director
   can parse.
2. **Optional stopping is fine.** Anytime-valid inference is what you want at
   $T = 14$; fixed-$n$ tests are not.
3. **It gets the epistemic-laundering argument exactly right.** A point-estimate
   ABM calibration implicitly claims its predictive is well-calibrated. The
   e-process is the bettor exploiting the miscalibration, with the bankroll
   quantifying how much the implicit claim is worth.

**Caveat.** Vovk's conformal prediction is a _different_ framework that is
frequently conflated with testing-by-betting. camdl teaches testing-by-betting
(Shafer 2021; Ramdas et al. 2023); conformal prediction is out of scope and may
appear as a separate module later.

### 6.4 User-interpretation caveat

The bankroll number is easy to misread. "$59.20 at α=0.05" is an objective
betting statistic, but a user who has not internalized what a supermartingale
is can report it as if it were a p-value (it isn't) or a Bayes factor (it
isn't, but closer). Two mitigations:

1. When `--show=betting` is invoked, `camdl compare` echoes a one-line
   interpretation disclaimer beneath the table (e.g. "bankroll numbers
   assume the baseline's predictive is calibrated; they are not p-values and
   not Bayes factors; see docs/betting.md for the interpretation contract").
2. The user-facing docs lead with Shafer (2021) *before* describing the
   subcommand flag, not the reverse. The betting framing is pedagogy, not a
   one-click feature.

This is a small intervention against a specific failure mode: a bankroll in a
paper defended by the phrase "it's what the tool said" with no accompanying
caveat about what the null is.

### 6.5 Recommended reading

- Shafer (2021), "Testing by betting: a strategy for statistical and scientific
  communication," _JRSS-A_.
- Ramdas, Grünwald, Vovk & Shafer (2023), "Game-theoretic statistics and safe
  anytime-valid inference," _Statistical Science_.

---

## 7. The `PrequentialTrace` type

Every inference pipeline produces a `PrequentialTrace` as a first-class output,
stored in the CAS entry alongside the posterior. Fields:

| Field            | Type                | Description                                                                |
| ---------------- | ------------------- | -------------------------------------------------------------------------- |
| `t0`             | `usize`             | Minimum window before scoring starts                                       |
| `times`          | `Vec<Time>`         | Time index for each scored step, length $T - t_0$                          |
| `y_obs`          | `Tensor`            | Observed value at each scored step                                         |
| `y_pred_samples` | `Tensor`            | Predictive samples (shape: $[T - t_0, S, d]$, with $d=1$ in v1)            |
| `log_score`      | `Vec<f64>`          | Per-step log score $\log \hat p(y_{t+1}\mid y_{1:t})$                      |
| `crps`           | `Vec<f64>`          | Per-step CRPS                                                              |
| `pit`            | `Vec<f64>`          | Per-step PIT value $u_t$                                                   |
| `ess`            | `Vec<f64>`          | Per-step filter ESS                                                        |
| `provenance`     | `enum`              | `PlugIn` in v1; Part II adds `LfoPsis`, `FullyBayesian`, `PseudoPosterior` |
| `warnings`       | `Vec<Warning>`      | Degenerate filter, under-identified $t_0$, etc.                            |

Part II adds: `psis_k_hat: Option<Vec<f64>>`, `refit_flags: Option<Vec<bool>>`,
and randomized-PIT seed fields. They are all `Option` so the v1 schema is a
forward-compatible prefix of the v2 schema — no migration.

Because per-step values are stored, all downstream summaries (total elpd, mean
CRPS, PIT histogram, sharpness, coverage, paired $\Delta$elpd with correct SE)
are computed without re-running the filter.

### 7.1 API surface

```python
result = camdl.fit(model, data, method="pmmh")
result.prequential              # PrequentialTrace
result.prequential.elpd()       # total log score, scalar
result.prequential.crps()       # mean CRPS, scalar
result.prequential.pit_coverage(level=0.9)   # scalar
result.prequential.pit_histogram(bins=10)    # returns binned counts
# Plotting lives in the camdl_diag Python package, not in camdl itself.
# `camdl compare --format json` emits the structured document plotting
# consumes — see §8.6.
```

#### Default `t0`

$t_0$ selection is load-bearing: scores made against an unidentified MLE are
dominated by initialization variance and lose power to rank models. The
default is tiered:

**Tier 0 (this release, default): model-structure heuristic with warning.**
$t_0$ defaults to a value derived from the model's structural identifiability
(e.g., "3–5 rising-limb observations" for SIR-type). This is a *guess about
model class*, not a measurement of the current fit, and the fit log says so
explicitly:

```
note: using structural-heuristic t0=5 for SIR-type model.
      This is not data-derived; the current fit's identifiability has
      not been verified. Run with --compute-t0-threshold to check.
```

User-supplied `t0` values below the structural heuristic emit a warning; below
1 they error.

**Tier 1 (opt-in, `--compute-t0-threshold`): identifiability sweep.** The
sweep runs the MLE pipeline on $y_{1:t}$ for increasing $t$ from multiple
random starts, and selects the smallest $t$ at which

1. $\|\hat\theta_t - \hat\theta_{t+1}\|$ falls below tolerance on the
   appropriate scale (the MLE has stabilized as more data arrives), AND
2. The between-start spread of $\hat\theta_t$ falls below tolerance (the
   optimum is identified, not just re-approached from similar initializations).

Both conditions must hold. (a) alone is defeated by correlated initialization;
between-start spread is the operational detector of non-identifiability. The
model-structure heuristic is used only as an *initial guess for the sweep
range*; it is not defensible as a definition of $t_0$ because it describes
identifiability in principle, not of the current fit.

Under `--compute-t0-threshold`, a user-supplied `t0` below the computed
threshold **errors** (not warns); overriding requires
`--allow-underidentified-t0`. If the MLE trajectory $\hat\theta_t$ has no
plateau over $t\in[1,T]$, the fit is reported as unidentified and no
prequential score is computed.

**Tier 2 (later release, default-to-Tier-1):** Once incremental IF2 caching
lets the sweep warm-start each $t$ from the $t-1$ fit, the sweep becomes
cheap enough to default to. Part II §13 spec'ing.

The Tier-0 warning is the minimum defensible discipline for v1. The Tier-1
sweep catches the class of failure where 0.2 nats of training-ll difference
amplifies into 50+ nats of holdout-ll difference from unidentified nuisance
parameters — a failure mode we characterized in testing during the
boarding-school rising-edge holdout experiment.

**Cost model note.** The Tier-1 sweep with $T$ fit points and $M$ starts is
$TM$ additional IF2 runs. For small problems ($T=14$, $M=4$) this is cheap.
For large problems ($T \approx 500$) it is prohibitive; users want
`--t0 <value>` with an explicit choice instead. That is fine: the point is
to refuse silent defaults that are hard to defend post-hoc, not to force the
sweep.

### 7.2 Pipeline notes (v1)

- **IF2 / MLE (plug-in).** `provenance = PlugIn`. Log score and CRPS computed
  directly from filter weights at $\hat\theta$. Fast, but scores should be
  understood as conditional on a point estimate.

Part II (§10–11) extends this to `PseudoPosterior` (from the IF2 chain cloud),
`LfoPsis` (for PMMH/PGAS with PSIS reweighting), and `FullyBayesian`
(full refit at each $t$).

### 7.3 What gets persisted to CAS

The trace is content-addressed and stored adjacent to the fit artifact.
Re-running `camdl fit` on the same model/data/method returns the cached trace.
`camdl prequential <ref>` exists to recompute a trace for a legacy fit that does
not have one; it is **never** invoked automatically by `camdl compare`.

---

## 8. Command-line UX: `camdl compare`

### 8.1 Shape

```
camdl compare <ref1> <ref2> [<ref3> ...] \
    [--metric elpd,crps,pit] \
    [--baseline <ref>]                # default: argmax elpd among refs
    [--show standard|betting|both]    # add/remove e-process columns
    [--format table|json|md] \
    [--preset <name>]
```

`<ref>` resolves in this order: CAS hash prefix (git-style, 4+ characters),
user-assigned name (via `camdl name <hash> <name>` or `[fit.name]` in
`fit.toml`), or path to a fit directory. Ambiguous prefixes fail loud; names are
per-CAS; overwrites require `--force`.

`--baseline` sets the reference model against which $\Delta$elpd (and,
optionally, the e-process) is computed. Default is the highest-elpd model
("best-baseline"); set explicitly to any `<ref>` for scientific comparisons
against a deliberate reference (e.g., a deterministic model when the question
is "does stochasticity earn its keep"). This matches the convention of
`loo_compare` in R, which users coming from brms/rstanarm will recognize.

There is **no separate pairwise subcommand.** The e-process / betting framing
(see §6) is available on the standard comparison table by passing
`--show=betting`; it adds columns (cumulative log-bankroll, first-crossing
step) without changing the command path. See §8.2 for the display.

### 8.2 Default table (multi-model, $\geq 2$ models)

```
$ camdl compare he2010-det he2010-stoch he2010-stoch-overdisp \
      --metric elpd,crps --baseline he2010-det

Model                    Method    T_score  elpd     Δelpd    se(Δ)   crps    Δcrps   PIT_cov90
─────────────────────────────────────────────────────────────────────────────────────────────
he2010-det               mle          9    -34.2      —        —      2.81     —       0.56
he2010-stoch             pmmh         9    -28.7    +5.5      2.1     2.14   -0.67     0.89
he2010-stoch-overdisp    pgas         9    -27.9    +6.3      2.3     2.09   -0.72     0.91

Scored on days 5–13 (t0=5, structural-heuristic default). Baseline: he2010-det.
⚠ he2010-det: PIT 90%-coverage 0.56 (nominal 0.90) — likely overconfident.
```

Design choices made deliberately:

- **`T_score` column is mandatory.** Users will inevitably compare fits with
  different `t0` or different data subsets. If `T_score` differs across rows,
  $\Delta$ columns are rendered as `—` (not computed) and a warning is printed.
  Silent incommensurability is the failure mode we aggressively refuse to
  tolerate. Override with `--allow-mismatched-horizon` and the user owns the
  result.
- **Paired $\Delta$ SEs.** Computed from pointwise differences, not independent
  totals:
  $$\text{se}(\Delta\text{elpd}) = \sqrt{T \cdot \text{Var}_t(\ell_t^A - \ell_t^B)}.$$
  This is the Vehtari–Gelman–Gabry standard and is typically substantially
  tighter than the naive independent-totals SE. Because per-step scores are
  stored in `PrequentialTrace`, this is a one-liner.
- **PIT coverage inline.** Not buried in a diagnostic subcommand. At small $T$,
  a miscalibrated model can win on mean elpd by getting lucky on a few days; the
  PIT column is the tell.
- **Warnings inline.** PSIS divergence, $t_0$ mismatch, different observation
  models, different data — all loud, at the bottom of the table, not in a log
  file.
- **No auto-annotated winner.** Propriety of the scoring rule does not rescue
  $\Delta\text{elpd} = 5.5 \pm 2.1$ at $T=9$ from being ambiguous. The table's
  job is to show the uncertainty and let the user (or reviewer) adjudicate.
- **No default model weights / stacking.** Stacking is a separate command
  (`camdl stack`) that users opt into when the workflow is ensemble forecasting.
  Putting weights in the default comparison table silently implies an ensemble
  workflow that most users do not want.

### 8.3 Betting display mode (`--show=betting`)

Invoking `camdl compare <refs...> --show=betting` adds e-process columns to
the standard table. No separate subcommand:

```
$ camdl compare he2010-det he2010-stoch --show=betting

Model            Method    T_score  elpd    Δelpd    log E_T    E_T     First α=0.05
────────────────────────────────────────────────────────────────────────────────────
he2010-det       mle          9    -34.2      —        —         —         —
he2010-stoch     pmmh         9    -28.7    +5.5     +4.08      59.2     t=8

ⓘ Bankroll numbers assume he2010-det's predictive is calibrated. They are
  not p-values and not Bayes factors. See docs/betting.md for the
  interpretation contract.
```

`log E_T` = $\sum_t \Delta\ell_t$ (cumulative log-bankroll), $E_T = \exp(\log E_T)$,
"First α=0.05" = smallest $t$ where $\mathcal{E}_t \geq 20$ (first crossing of
the Ville bound at level 0.05). The interpretation disclaimer is emitted
automatically (§6.4).

### 8.4 Structural-fairness preflight

Before rendering, `compare` refuses comparisons that are structurally unfair.
Each refusal is a hard error (not a footnote) with an explicit override flag:

| Condition | Override |
|---|---|
| Different `data_hash` across fits (data was edited between runs) | `--allow-data-mismatch` |
| Different observation models (e.g. Poisson vs continuous density) | `--allow-obs-model-mismatch` |
| Different particle counts (PF noise is differently biased) | `--allow-particle-mismatch` |
| Different backends or `dt` | `--allow-backend-mismatch` |
| `T_score` differs across rows (already in §8.2, override: `--allow-mismatched-horizon`) | `--allow-mismatched-horizon` |

Silent comparison of apples to oranges is the class of failure that produces
an unreproducible paper table defended by "it's what the tool said." The
override flags are deliberately verbose so they show up in a command history
or in a shell-history review.

### 8.5 Anti-pattern detection

`compare` also flags common user mistakes:

| Pattern | Action |
|---|---|
| `t0 = 0` (no burn-in) | error; require `--allow-no-burnin` |
| All models have identical elpd to 3 decimals | error (probable seed / cache collision) |
| Same seed across models but different `n_particles` | warn (paired SE biased downward) |
| One model's MLE converged at a parameter bound, others did not | warn per-parameter |
| User-supplied `t0` below the structural-heuristic default | warn |
| User-supplied `t0` below `--compute-t0-threshold` result | error; override `--allow-underidentified-t0` |

These are things the tool can learn once so users don't have to.

### 8.6 Preset comparisons in `fit.toml`

For reproducibility (the paper table should regenerate from one command),
`fit.toml` accepts a `[compare]` section:

```toml
[compare.main]
models = ["he2010-det", "he2010-stoch", "he2010-stoch-overdisp"]
baseline = "he2010-det"
metrics = ["elpd", "crps", "pit_cov90"]
format = "md"

[compare.betting_main]
models = ["he2010-det", "he2010-stoch"]
baseline = "he2010-det"
show = "betting"
alpha = 0.05
```

Then `camdl compare --preset main` and `camdl compare --preset betting_main`
run the canonical comparisons. Presets live in the repo next to the models.

### 8.7 CAS integration: strict non-recomputation

`compare` never silently recomputes. Each fit's `PrequentialTrace` is
content-addressed and stored in CAS alongside the fit. `compare` fetches and
joins. If a fit was saved without a prequential trace (legacy fit), the command
fails loudly with `camdl prequential <ref>` as the suggested fix.
Auto-computation on comparison is rejected because subcommands with hidden side
effects are the class of bug that produces unreproducible paper tables.

### 8.8 Output formats — plotting lives in `camdl_diag`

- `--format table` (default): ANSI-colored console table.
- `--format md`: GitHub-flavored markdown table, drop-in for papers and PRs.
- `--format json`: full structured output for programmatic use and plotting.

**Plotting is explicitly not a `compare` subcommand.** The Rust CLI produces
computation and structured text; figure rendering happens in the companion
Python package `camdl_diag`, which consumes `--format json` and renders with
the book's matplotlib conventions. This separation has three benefits:

1. **Keeps the CLI minimal.** No matplotlib dependency in the Rust binary;
   no opinionated default styling that users have to fight.
2. **Reproducibility without tool.** The JSON is content-addressable and
   archivable; figures regenerate from the JSON without a camdl binary
   present.
3. **Matches the existing split.** Book repo already has `styles.py`,
   `profile_plot.py`, `plot_traces.py` as proto-`camdl_diag`.

The JSON schema is therefore **the plotting contract** and must be rich
enough for downstream figures: per-step log score and CRPS arrays, per-step
PIT values, per-step ESS, per-step cumulative e-process (if `--show=betting`
was requested), full warning list, and a reference to the
`PrequentialTrace`'s `y_pred_samples` tensor in CAS (so the Python side can
compute custom scoring rules on the same particles without re-running the
PF). The schema is versioned and non-breaking within a major release.

All formats include the warnings and the scoring window in a machine-readable
form so that downstream tools (including the white-paper build) cannot
silently drop them.

---

## 9. Worked example: He et al. (2010) boarding school

The canonical test case. Settings:

- `t0 = 5` (full rising limb + peak; $R_0$ identified).
- 9 one-step-ahead scores, days 5–13.
- Models compared:
  1. `he2010-det` — deterministic SIR with Poisson observations, MLE.
  2. `he2010-stoch` — Euler-Maruyama stochastic SIR, PMMH.
  3. `he2010-stoch-overdisp` — stochastic SIR with negative-binomial
     observations, PGAS/NUTS.

This is the table users see, and the table in the calibration methodology white
paper. See §8.2 for the rendered output.

**What the table is expected to show.** The deterministic model interpolates the
mean trajectory well (CRPS is not catastrophic) but is badly overconfident (PIT
coverage ≈ 0.56 against a nominal 0.9). The stochastic models pay a small
mean-CRPS cost for substantially better calibration. The pairwise e-process
against the deterministic baseline crosses $1/\alpha = 20$ well inside the 9
observations — quantifying, in betting terms, exactly how much the
stochastic-observation admission is worth.

**What the table is not.** It is not a licence to claim "the stochastic model is
correct." With $T = 9$ scored points the comparison is directional evidence, not
proof. The report's job is to make this legible. The e-value reading — "$59 of
evidence at α=0.05" — is honest about scale in a way that a bare p-value is not.

---

# Part II — v2 scope (later release)

The items below are committed design, deferred implementation. They do not
affect v1 semantics; the v1 `PrequentialTrace` schema is a forward-compatible
prefix of the v2 schema (new fields are `Option<T>` where absence means
"v1 trace"). The v1 `provenance = PlugIn` value is preserved verbatim; Part II
adds enum variants but does not change existing ones.

## 10. Fully Bayesian predictive (LFO-PSIS)

For fully Bayesian pipelines, the $\theta$ integral is nontrivial: the
predictive at time $t$ requires the _partial-data_ posterior
$p(\theta \mid y_{1:t})$, not the full-data posterior $p(\theta \mid y_{1:T})$
produced by the MCMC run. Naively this requires refitting MCMC at every $t$ —
for boarding school, 9–10 refits.

v2 implements the approximate leave-future-out cross-validation (LFO-CV) of
Bürkner, Gabry & Vehtari (2020): treat the full-data posterior
$\pi(\theta \mid y_{1:T})$ as a reference proposal and importance-reweight
backward to any earlier partial-data posterior,

$$
\pi(\theta \mid y_{1:t}) \propto \pi(\theta \mid y_{1:T}) \cdot \frac{1}{p(y_{t+1:T} \mid \theta, y_{1:t})},
$$

giving weights $w^{(s)} \propto 1/p(y_{t+1:T} \mid \theta^{(s)}, y_{1:t})$ for
posterior draws $\theta^{(s)} \sim \pi(\theta \mid y_{1:T})$. Weights are
stabilized with Pareto-smoothed importance sampling (PSIS; Vehtari et al.
2024). When $\hat k > 0.7$ at some $t$, camdl refits MCMC at that $t$ and
resets the reference distribution. Typically cuts 10 refits to 2–3.

For PMMH and PGAS specifically, the partial-data likelihood
$p(y_{t+1:T}\mid\theta^{(s)}, y_{1:t})$ is a bootstrap-filter replay on each
posterior draw — no new MCMC chain per posterior sample. `provenance`
variants `LfoPsis` and (for budget-exhausted cases where all refits
complete) `FullyBayesian` distinguish the execution path.

Pipeline notes:

- **PMMH, CPM-PMMH.** `provenance = LfoPsis` by default.
- **PGAS / NUTS.** Same as PMMH. Because PGAS mixes in $\theta$-space better
  than PMMH past $K \approx 600$ observations, the PSIS diagnostics look
  better at equivalent compute.

## 11. Pseudo-posterior from IF2 cloud

IF2 under cooling produces a final-iteration chain distribution that is a
*rough* approximation to the posterior. Calling it `plug_in` (the v1 default)
throws information away: the final particle cloud across chains has
meaningful spread that captures parameter uncertainty, even though it is not
a well-calibrated posterior.

v2 offers this as a third provenance variant `PseudoPosterior`, selected by
`camdl fit --pseudo-posterior` or `camdl prequential --pseudo-posterior`.
LFO-PSIS runs against the cloud as if it were a posterior, with a standing
caveat in the docs ("uncalibrated posterior approximation; use when PMMH is
infeasible; report IF2 + LFO-PSIS scores as such"). Strictly better than
plug-in for the common case where PMMH is too expensive.

The provenance label forces the caveat to travel with the score.

## 12. Randomized PIT for discrete observations

For count data (the dominant case in epidemic applications), point-estimate
PIT exhibits stair-step artifacts near integer values. The randomized PIT of
Czado, Gneiting & Held (2009) corrects this:

$$
u_t = F_{t+1\mid t}(y_{t+1} - 1) + v_t\big[F_{t+1\mid t}(y_{t+1}) - F_{t+1\mid t}(y_{t+1} - 1)\big], \quad v_t \sim \text{Uniform}(0,1),
$$

which recovers uniformity under correct calibration of the discrete
predictive. v2 adds a `randomization_seed: Option<u64>` field to
`PrequentialTrace` so the randomization is reproducible; v1 simply uses
point-estimate PIT with a documented stair-step caveat.

## 13. Identifiability sweep (Tier-1 $t_0$ default)

v1 defaults $t_0$ to a model-structure heuristic with a warning (§7.1 Tier 0)
and offers `--compute-t0-threshold` as an opt-in. v2 makes the sweep the
default once incremental IF2 caching (warm-starting each $t$ from the $t-1$
fit's MLE) has made the cost tractable:

Under Tier 1 as default, a user-supplied `t0` below the computed threshold
errors; override requires `--allow-underidentified-t0`. If the MLE trajectory
$\hat\theta_t$ has no plateau over $t\in[1,T]$, the fit is reported as
unidentified and no prequential score is computed.

## 14. PSIS $\hat k$ diagnostic

LFO-specific. Thresholds follow Vehtari et al. (2024):

- $\hat k \leq 0.7$: importance weights reliable, reweight is used.
- $\hat k > 0.7$: refit triggered; reference distribution reset; trace
  records the refit.

**Silent fallback is forbidden.** If refitting fails (e.g. budget exhausted),
the trace records a failure flag at that $t$ and `camdl compare` refuses to
render the score without `--allow-degenerate`. The anti-pattern we are
avoiding: silently falling back to the full-data $\theta$ posterior, which
gives an optimistic score and a user who does not know it.

v2 adds fields `psis_k_hat: Option<Vec<f64>>` and
`refit_flags: Option<Vec<bool>>` to `PrequentialTrace`. v1 traces have both
as `None`.

## 15. Rolling-origin $k$-step-ahead

For nowcasting workflows where the decision horizon is $k > 1$, v2 supports
`--horizon k` in `fit` and `prequential`; the filter propagates $k$ steps
without assimilating before the predictive is emitted. The score is then
$\sum_t \log p(y_{t+k}\mid y_{1:t})$. The `compare` table UX is unchanged
except that `T_score` adjusts to the reduced count of $k$-step predictions.

Note that the e-value / betting interpretation in §6 remains valid for
$k$-step predictives — the supermartingale property holds — but the bets
being made are block-predictive, not step-predictive. Worth explicit mention
in the v2 user docs.

## 16. Energy score and panel / spatial comparison

For multivariate predictives (e.g. Nigeria LGA panels), CRPS generalizes to
the energy score (Gneiting & Raftery 2007),

$$
\text{ES}(F, y) = \mathbb{E}_F\|Y - y\| - \tfrac{1}{2}\mathbb{E}_F\|Y - Y'\|,
$$

which handles cross-unit dependence natively. Summing univariate CRPS across
LGAs would discard exactly the spatial correlation structure the model is
meant to capture. `PrequentialTrace`'s `y_pred_samples` tensor is defined in
v1 with shape $[T - t_0, S, d]$ already ($d=1$ in v1, $d > 1$ in v2) so
energy score drops in without schema migration.

The v2 `compare` UX for multi-unit data — whether per-unit scores are
summarized with a scalar, a per-unit table, or a small-multiples panel — is
an open question to be resolved with a dedicated design round.

## 17. Other deferred items

**Two-window / train-gap-test.** Available as a non-default split strategy
for users who specifically want it, but not recommended for single-outbreak
data: a train-gap-test split introduces an unmotivated discontinuity in a
trend-dominated series. Its use case is serial-autocorrelation leakage in
stationary series, which is not the setting for most compartmental models.

**Stacking / model averaging.** Separate command `camdl stack`. Intentionally
not a default output of `compare` in either v1 or v2. Users who want an
ensemble workflow opt in explicitly.

**Conformal prediction.** A *different* framework from testing-by-betting
despite being frequently conflated with it. A separate module may appear
once the core prequential UX stabilizes; out of scope for v2.

---

# Part III — cross-cutting

## 18. Honest limitations

At $n \approx 14$ with a single trajectory, no out-of-sample procedure yields
sharp model selection. The prequential framework yields the _right kind of
uncertainty_ (per-step score variance, PIT calibration, anytime-valid e-values)
rather than disappearing the uncertainty into a single scalar. For the
epistemic-laundering argument, the defensible position on datasets this size is
the combination of:

- Rolling-origin scoring with a proper scoring rule (this document).
- Parameter recovery on simulated replicates at matched sample size (camdl's
  `simulate_and_recover`).
- Posterior predictive checks on the full trajectory (camdl's `ppc`).

A single post-hoc train/test split is a gesture at validation, not a
measurement, and is labeled as such in the white paper.

---

## 19. References

- Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of
  cross-validation for evaluating autoregressive time series prediction.
  _Computational Statistics & Data Analysis_, 120, 70–83.
- Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating
  epidemic forecasts in an interval format. _PLoS Computational Biology_, 17(2),
  e1008618.
- Bürkner, P.-C., Gabry, J., & Vehtari, A. (2020). Approximate leave-future-out
  cross-validation for Bayesian time series models. _Statistics and Computing_,
  30, 1297–1319.
- Czado, C., Gneiting, T., & Held, L. (2009). Predictive model assessment for
  count data. _Biometrics_, 65(4), 1254–1261.
- Dawid, A. P. (1984). Statistical theory: the prequential approach. _Journal of
  the Royal Statistical Society: Series A_, 147(2), 278–292.
- Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic
  forecasts, calibration and sharpness. _Journal of the Royal Statistical
  Society: Series B_, 69(2), 243–268.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. _Journal of the American Statistical Association_,
  102(477), 359–378.
- He, D., Ionides, E. L., & King, A. A. (2010). Plug-and-play inference for
  disease dynamics: measles in large and small populations as a case study.
  _Journal of the Royal Society Interface_, 7(43), 271–283.
- Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and
  Practice_ (3rd ed.). OTexts. https://otexts.com/fpp3/
- Laio, F., & Tamea, S. (2007). Verification tools for probabilistic forecasts
  of continuous hydrological variables. _Hydrology and Earth System Sciences_,
  11(4), 1267–1277.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic
  statistics and safe anytime-valid inference. _Statistical Science_, 38(4),
  576–601.
- Shafer, G. (2021). Testing by betting: a strategy for statistical and
  scientific communication. _Journal of the Royal Statistical Society: Series
  A_, 184(2), 407–431.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an
  analysis and review. _International Journal of Forecasting_, 16(4), 437–450.
- Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto
  smoothed importance sampling. _Journal of Machine Learning Research_, 25(72),
  1–58.
