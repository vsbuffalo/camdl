# Model comparison in camdl: what `compare` computes, and how to read it

**Scope:** every number, label, and caveat that `camdl compare` prints — the
prequential score it is built on, the two ways a score can be optimistic and how
camdl labels them, the scoring rules, the uncertainty attached to a difference,
and the held-out evaluation mode. This is the page the table's footer points at.

**Authoritative code:** the scoring kernels live in
`rust/crates/sim/src/inference/prequential.rs`, the evidence formatting in
`rust/crates/cli/src/evidence.rs`, and the comparison logic (preflights,
derivation, rendering) in `rust/crates/cli/src/compare.rs`. The design record is
`docs/dev/proposals/2026-08-29-honest-predictive-evaluation.md`.

---

## 1. The object being scored: one-step-ahead predictions

A fitted model is, among other things, a machine for making probabilistic
forecasts: given everything observed so far, it assigns a probability
distribution to the next observation. Comparing models by the quality of these
distributions — rather than by how closely a fitted curve threads the data — is
the prequential ("predictive sequential") approach of Dawid (1984), and it is
what `camdl compare` does.

Write $y_{1:T}$ for the observation series and $\theta$ for the model
parameters. At each step the model's one-step-ahead predictive density is

$$p(y_t \mid y_{1:t-1}, \theta),$$

read as: the probability the model assigns to the value actually observed at
step $t$, having seen only the observations before $t$. For a state-space model
this quantity is computed by the particle filter as a by-product of filtering —
the filter's particle cloud at step $t$ _is_ the model's belief about the latent
state given $y_{1:t-1}$, and averaging the observation density over that cloud
gives the predictive.

The headline score is the **expected log predictive density** (elpd): the sum of
log predictive densities over the scored steps,

$$\mathrm{elpd} \;=\; \sum_{t \in \text{scored}} \log \hat p(y_t \mid y_{1:t-1}, \theta).$$

Larger (less negative) is better. The log of a density is used, rather than the
density itself, because the log score is a _strictly proper_ scoring rule
(Gneiting & Raftery 2007): a forecaster maximizes its expected value only by
reporting its true predictive distribution, so the score cannot be gamed by
hedging.

One identity is worth internalizing, because it is the reason this page keeps
insisting on labels. If $\theta$ was fit to the **full** series and **every**
step is scored, the sum telescopes:

$$\sum_{t=1}^{T} \log \hat p(y_t \mid y_{1:t-1}, \hat\theta) \;=\; \log \hat p(y_{1:T} \mid \hat\theta),$$

which is exactly the in-sample log-likelihood at the fitted point. A
"prequential" score under those conditions is one-step-ahead in $y$ but not in
$\theta$ — it has already seen the future through the parameters. Whether a
score has this character is not something you should have to infer; camdl stamps
it (§2).

## 2. The two ways a score can flatter, and the two stamps

Every prequential trace carries two typed labels, printed in the JSON output and
governing the caveats under the table.

**Conditioning — which data $\theta$ has seen.**

- `in_sample`: $\theta$ was fit to all of the scored data. By the identity
  above, the elpd is the in-sample log-likelihood. Crucially, the optimism does
  **not** cancel when two models are differenced: it grows with the effective
  number of parameters each $\theta$ was free to tune (the bias term that
  AIC-style corrections estimate), so an in-sample Δelpd tilts toward the more
  flexible model. The caveat under the table says this whenever it applies.
- `hold_out_tail`: $\theta$ comes from a fit that was sealed at a boundary
  $\tau$ — its training data end there — and only observations with $t > \tau$
  are scored. The filter still conditions on _all_ past observations as it
  scores (assimilating each held-out week after scoring it); that is legitimate,
  because forecasting from the past is the task. Only $\theta$ must not have
  seen the future, and here it provably has not (§6.3).

**Provenance — how $\theta$ enters the predictive.**

- `plug_in`: scored at a single parameter value (an optimizer's winner, or a
  posterior mean). Plug-in predictives drop parameter uncertainty, so they are
  sharper than the model honestly is — typically visible as too-narrow intervals
  and low PIT coverage (§4.3).
- `posterior`: the predictive is a mixture over $M$ draws from the posterior
  (§5). This is the honest Bayesian predictive; it widens the intervals by
  exactly the parameter uncertainty the plug-in ignores.

The axes are independent: a posterior-mixture score can still be in-sample, and
a held-out score can still be plug-in. `compare` refuses to difference rows
whose conditioning differs — the two quantities answer different questions — and
mixing provenance requires an explicit flag.

## 3. Reading the table, column by column

```
Model      T_score     elpd   Δelpd       LR   se(Δ)               evidence    crps    Δcrps   PIT_cov90
```

- **T_score** — the number of scored observations. Comparisons demand the same
  scored set on every row; mismatched horizons, mismatched observation times,
  and mismatched stream sets are all refused before rendering.
- **elpd** — the summed log predictive density (§1), in nats (natural-log
  units). Its absolute value is not interpretable across datasets; only
  differences on the same scored set are.
- **Δelpd** — this row's elpd minus the baseline's, computed pointwise on the
  paired per-step scores. Positive favors this row.
- **LR** — $e^{\Delta\mathrm{elpd}}$, the likelihood ratio against the baseline:
  how many times more probability this model assigned to the scored
  observations. It is _not_ a Bayes factor (no integration over $\theta$) and
  _not_ an e-value (no anytime-valid guarantee) — it is the exponentiated score
  difference, nothing more.
- **se(Δ)** — the standard error of Δelpd (§4.4). Read it as descriptive.
- **evidence** — the difference restated in decibans with a qualitative tier
  (§4.5), or `within noise` when the difference is inside two standard errors.
- **crps / Δcrps** — the mean continuous ranked probability score (§4.2), in the
  units of the observations, and its per-step mean difference vs the baseline.
  Lower is better.
- **PIT_cov90** — the fraction of scored observations that fell inside the
  model's central 90% predictive interval (§4.3). Nominal is 0.90; well below it
  means overconfident intervals.

## 4. The scoring rules and diagnostics

### 4.1 Log score

Defined in §1. Two properties matter in practice. It is _local_: only the
density at the realized value counts, so a model is rewarded for putting mass
exactly where the data landed. And it is harsh on tail failures: an observation
the model considered nearly impossible contributes a very large negative term.
That harshness is the rule working as designed, but at small $T$ it means one
bad week can dominate the sum — which is why CRPS is reported alongside.

### 4.2 CRPS

The continuous ranked probability score of a predictive distribution $F$ against
an observation $y$ is

$$\mathrm{CRPS}(F, y) \;=\; \mathbb{E}_F\lvert X - y\rvert \;-\; \tfrac{1}{2}\,\mathbb{E}_F\lvert X - X'\rvert,$$

with $X, X'$ independent draws from $F$. The first term rewards putting draws
near the observation; the second penalizes diffuseness, which is what makes the
rule proper (Gneiting & Raftery 2007). CRPS is in the units of $y$ ("model A
beats B by 0.8 cases per week"), degrades gracefully in the tails, and is
therefore the robust companion to the log score: when the two rules disagree
about a ranking, the disagreement is usually a tail-calibration finding, not
noise.

From $S$ predictive samples camdl uses the _fair_ estimator (Ferro 2014;
pairwise form as in Zamo & Naveau 2018),

$$\widehat{\mathrm{CRPS}} \;=\; \frac{1}{S}\sum_{s}\lvert x_s - y\rvert \;-\; \frac{1}{2S(S-1)}\sum_{s \ne s'}\lvert x_s - x_{s'}\rvert,$$

which is unbiased for the CRPS of the underlying predictive at finite ensemble
size — so scores computed at different sample counts remain comparable. Both
this and the log score are validated against the `scoringRules` R package with
committed reference cases.

### 4.3 PIT and interval coverage

The probability integral transform asks: where did the observation fall within
the predictive distribution? For a continuous predictive, $u_t = F_t(y_t)$ is
uniform on $[0,1]$ exactly when the predictive is calibrated (Dawid 1984;
Gneiting, Balabdaoui & Raftery 2007). Clumping near $0$ and $1$ means the
observations keep landing in the tails — the intervals are too narrow.

Count data need one adjustment: with a discrete predictive, $F_t(y_t)$ takes
only a few values and is not uniform even for a perfect model. camdl uses the
**randomized PIT**,

$$u_t \;=\; \hat P(X < y_t) \;+\; v_t\,\hat P(X = y_t), \qquad v_t \sim \mathrm{Uniform}(0,1),$$

which spreads each observation uniformly across its own probability atom and
restores exact uniformity under calibration. The randomized transform is due to
Smith (1985) and Brockwell (2007); Czado, Gneiting & Held (2009) is the standard
reference for predictive assessment of count data (their own proposal is a
nonrandomized variant). The randomization seed is recorded in the trace, so the
numbers are reproducible.

`PIT_cov90` compresses this to one number: the fraction of $u_t$ in
$[0.05, 0.95]$. A plug-in predictive commonly shows coverage well below nominal
— that is the parameter uncertainty it dropped, and the posterior-mixture
provenance (§5) is the remedy.

### 4.4 What se(Δ) is, and what it is not

The paired standard error of Δelpd treats the per-step differences
$d_t = \ell^A_t - \ell^B_t$ as the data:

$$\mathrm{se}(\Delta) \;=\; \sqrt{T \cdot \widehat{\mathrm{Var}}_t(d_t)}$$

(Vehtari, Gelman & Gabry 2017). Two refinements and one honest limitation:

- **Serial dependence.** Score differences along a time series are
  autocorrelated precisely when a model is misspecified, which makes the naive
  variance too small. camdl uses a heteroskedasticity-and-
  autocorrelation-consistent (Newey & West 1987) long-run variance with the
  Harvey, Leybourne & Newbold (1997) small-sample correction — the standard
  treatment in forecast comparison since Diebold & Mariano (1995). The lag-one
  autocorrelation of $d_t$ is printed so you can see how much this mattered.
- **Filter noise.** Each elpd is itself a Monte Carlo estimate from a particle
  filter, and its bias is model-dependent (the log of an unbiased estimate is
  biased downward, by more for models whose filters have higher weight variance;
  Bérard, Del Moral & Doucet 2014). camdl therefore derives each score as $R$
  replicate filter runs combined by log-mean-exp — the practice pomp established
  (King, Nguyen & Ionides 2016) — and prints the resulting MC standard error. A
  Δelpd within twice the filter noise is labelled as such.
- **The limitation:** all of this rests on a normal approximation that is
  unreliable when the scored count is small (tens rather than hundreds) or the
  models predict alike (Sivula, Magnusson, Matamoros & Vehtari,
  arXiv:2008.10296). Compartmental fits routinely score 10–50 steps. The table
  says so whenever it applies; read se(Δ) as a description of scatter, not as a
  hypothesis test.

### 4.5 The evidence column: decibans and the Jeffreys tiers

A log-likelihood-ratio difference is easier to talk about in **decibans** (dB):
ten times the base-10 logarithm, so $+10$ dB means "assigned ten times the
probability," $+20$ dB a hundred times. One nat is $10 / \ln 10 \approx
4.34$ dB. The deciban as a unit of weight of evidence is due to Good (1950),
from wartime work with Turing.

The qualitative words attached to magnitudes follow Jeffreys (1961, Appendix B),
with one extension of camdl's own:

| band        | odds           | tier          | source          |
| ----------- | -------------- | ------------- | --------------- |
| 0–5 dB      | up to ~3:1     | indeterminate | Jeffreys (1961) |
| 5–10 dB     | ~3:1 to 10:1   | substantial   | Jeffreys (1961) |
| 10–15 dB    | 10:1 to ~30:1  | strong        | Jeffreys (1961) |
| 15–20 dB    | ~30:1 to 100:1 | very strong   | Jeffreys (1961) |
| 20–40 dB    | 100:1 to 10⁴:1 | decisive      | Jeffreys (1961) |
| above 40 dB | beyond 10⁴:1   | overwhelming  | camdl extension |

Jeffreys' own top tier is unbounded ("decisive, 20 dB and up"); camdl splits it
because epidemic-scale comparisons routinely produce thousands of decibans and
the single word erased that range. Kass & Raftery (1995) give the modern
alternative scale; its thresholds sit at different decibel values, so the two
scales are not interchangeable.

Two guards keep the words honest. First, a tier is printed only when
$\lvert\Delta\mathrm{elpd}\rvert \ge 2\,\mathrm{se}(\Delta)$ — otherwise the
cell reads `within noise`, because a tier names a magnitude the data cannot
resolve. Second, the footer states on every render that these scales calibrate
**Bayes factors** — ratios of marginal likelihoods, with $\theta$ integrated out
— and a camdl Δelpd is not yet one: at plug-in provenance it is a ratio of
maximized (or point-evaluated) likelihoods, which flatters the more flexible
model in a way a Bayes factor would not. Treat the tier as a readability aid for
the magnitude, not as a Bayesian conclusion.

## 5. The posterior-mixture predictive

For a Bayesian fit, scoring at a single point (even the posterior mean)
understates the model's honest uncertainty. The posterior predictive
marginalizes instead:

$$\hat p(y_t \mid y_{1:t-1}, \mathcal{D}) \;=\; \frac{1}{M}\sum_{m=1}^{M} \hat p(y_t \mid y_{1:t-1}, \theta_m), \qquad \theta_m \sim p(\theta \mid \mathcal{D}),$$

estimated by running one filter per draw and averaging the per-step _densities_
(a log-sum-exp over draws). Averaging the per-draw _log_ scores instead would
give a strictly lower, different number — that is Jensen's inequality, and the
distinction matters: the mixture scores the model's actual predictive
distribution, while the mean of logs scores each parameter value separately.
Predictive samples are pooled across the $M$ filters, so CRPS, PIT, and the
intervals all inherit parameter uncertainty too.

`compare` uses the mixture automatically for fits with a posterior draws cloud
(`--draws`, default 64, controls $M$; `--draws 1` falls back to the
posterior-mean plug-in). Chain exclusions (`--exclude-chains`) apply to the
cloud before drawing. The per-fit cost is $M \times R$ filter passes and is
printed before the derivation runs.

## 6. Held-out evaluation

### 6.1 Declaring a holdout

Two forms, mutually exclusive, both in `fit.toml`:

```toml
[data]
holdout_after = 42.0 # or a date, or "last_obs - 4 weeks"

[data.observations]
weekly_cases = "cases.tsv" # the FULL series; rows past the boundary
# are withheld from training automatically
```

or, when the split already exists as files (e.g. from `camdl data split`):

```toml
[data.observations]
weekly_cases = "cases_train.tsv"

[data.holdout]
weekly_cases = "cases_holdout.tsv"
```

Both are tail-only: every held-out time must lie strictly after the last
training time, and the fit refuses otherwise. Held-out points interleaved with
training points are a different procedure (block cross-validation) with
different semantics, and camdl will not silently blur the two.

### 6.2 What gets scored

When every compared fit declares a holdout, `compare` scores each model over the
full series at that model's sealed parameters, counting only the observations
past the boundary. Scoring is one-step-ahead with assimilation: the filter
forecasts a held-out week, is scored on it, and then updates its state on that
week before forecasting the next. Errors do not compound; each scored term is a
genuine forecast of an observation the parameters never saw. `--in-sample`
forces the old full-series in-sample comparison, clearly stamped.

### 6.3 The non-leakage guarantee, and its limit

"Held out" is verified, not asserted. A fit's identity seals the training data
(by content digest) and the holdout declaration; applying the split writes the
training window into the fit's metadata at the moment the rows are withheld.
`compare` refuses the `hold_out_tail` label unless the fit carries that
applied-window record, the declared window covers every scored time, and the
scored observations lie strictly past the training window. A fit produced by any
camdl version that did not enforce the split cannot carry the record, and is
refused the label.

The guarantee is exactly "no data leakage into $\theta$." It cannot certify that
the _modeler_ chose the model family without having seen the full curve — that
discipline is yours.

### 6.4 What a tail score can and cannot conclude

The held-out tail of an epidemic is usually the hard part — transmission is
changing there — so held-out scores are lower than in-sample intuition expects.
That is the deployment condition, not a bias: a forecast used for decisions is
always made at the frontier. In a comparison the shared difficulty largely
cancels, favoring (correctly) the model that tracks regime change. What a short
tail cannot give is power: with a handful of scored weeks, expect `within noise`
verdicts, and treat a decisive tier with corresponding respect. Scoring across
several origins (rolling-origin evaluation) is the planned remedy for both the
small count and the single-phase dependence of one split.

## 7. Practical guidance

- Prefer held-out comparison whenever the question is "which model should we
  believe going forward." Reserve `--in-sample` for exploratory triage, and read
  its verdicts knowing the flexible-model bias.
- Report the pair (Δelpd, se) and the CRPS alongside any tier word; if log score
  and CRPS disagree on a ranking, investigate tail calibration before choosing.
- Check `PIT_cov90` before trusting any interval you plan to show a decision
  maker; a plug-in score with low coverage is understating uncertainty by
  construction.
- The per-observation vector behind every Δelpd is available via `--pointwise` —
  "model B wins by 12 nats" and "model B wins entirely on three weeks around the
  intervention" are different findings, and the pointwise file distinguishes
  them.

## References

- Bérard, J., Del Moral, P., & Doucet, A. (2014). A lognormal central limit
  theorem for particle approximations of normalizing constants. _Electronic
  Journal of Probability_, 19(94).
- Brockwell, A. E. (2007). Universal residuals: a multivariate transformation.
  _Statistics & Probability Letters_, 77(14), 1473–1478.
- Czado, C., Gneiting, T., & Held, L. (2009). Predictive model assessment for
  count data. _Biometrics_, 65(4), 1254–1261.
- Dawid, A. P. (1984). Statistical theory: the prequential approach. _Journal of
  the Royal Statistical Society, Series A_, 147(2), 278–292.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
  _Journal of Business & Economic Statistics_, 13(3), 253–263.
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. _Quarterly Journal
  of the Royal Meteorological Society_, 140(683), 1917–1923.
- Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic
  forecasts, calibration and sharpness. _Journal of the Royal Statistical
  Society, Series B_, 69(2), 243–268.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. _Journal of the American Statistical Association_,
  102(477), 359–378.
- Good, I. J. (1950). _Probability and the Weighing of Evidence_. Griffin.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
  prediction mean squared errors. _International Journal of Forecasting_, 13(2),
  281–291.
- Jeffreys, H. (1961). _Theory of Probability_ (3rd ed.), Appendix B. Oxford
  University Press.
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. _Journal of the American
  Statistical Association_, 90(430), 773–795.
- King, A. A., Nguyen, D., & Ionides, E. L. (2016). Statistical inference for
  partially observed Markov processes via the R package pomp. _Journal of
  Statistical Software_, 69(12).
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  _Econometrica_, 55(3), 703–708.
- Sivula, T., Magnusson, M., Matamoros, A. A., & Vehtari, A. Uncertainty in
  Bayesian leave-one-out cross-validation based model comparison.
  arXiv:2008.10296.
- Smith, J. Q. (1985). Diagnostic checks of non-standard time series models.
  _Journal of Forecasting_, 4(3), 283–291.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
  evaluation using leave-one-out cross-validation and WAIC. _Statistics and
  Computing_, 27, 1413–1432.
- Zamo, M., & Naveau, P. (2018). Estimation of the continuous ranked probability
  score with limited information and applications to ensemble weather forecasts.
  _Mathematical Geosciences_, 50(2), 209–234.
