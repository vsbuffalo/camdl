---
status: proposal
date: 2026-08-29
supersedes: 2026-05-31-prequential-bayesian-lfo.md
---

# Honest predictive evaluation: trustworthy `compare`, holdout, and the posterior-mixture predictive

**Scope:** the `camdl compare` / prequential evaluation stack — what it
measures, how its uncertainty is quantified, and the staged path from today's
in-sample plug-in score to machine-verified out-of-sample evaluation.
**Supersedes** `2026-05-31-prequential-bayesian-lfo.md`: LFO-PSIS is carried
forward here as a named follow-up (§3.9) rather than the next increment.
**Audience:** camdl contributors; the methods section (§3) is written to be
liftable into the whitepaper and book.

---

## 0. Executive summary

`camdl compare` ranks models by a quantity labelled "prequential elpd." Because
the particle filter resamples at every observation and resets weights to
uniform, the per-step log score is identical to the filter's per-step likelihood
increment, so the trace's elpd telescopes to the in-sample conditional
log-likelihood at the plug-in parameter: exactly `log p̂(y_{1:T} | θ̂)`,
decomposed one step at a time. The machinery is exact and its kernels are
correct against the literature; the problem is classification, not computation.
Three consequences follow: the Δelpd between models carries no complexity
penalty (it is biased toward the more flexible model); the `E_T` column and
Jeffreys evidence labels borrow guarantee vocabulary (e-values, Bayes-factor
scales) that the in-sample quantity does not earn; and there is no honest
out-of-sample mode at all — the fit-config holdout declarations
(`holdout_after`, `[data.holdout]`) are parsed and hashed into run identity but
never applied (gh#585).

This proposal fixes the vocabulary now, then builds the honest modes, in five
stages. Each stage is a breakpoint: it lands as a reviewable unit and leaves
`compare` strictly more trustworthy than before it.

| Stage | Lands                                                                                                                | Breakpoint promise                                                                                |
| ----- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 1     | Relabels (`E_T` → `LR`), evidence gated on `se(Δ)`, warnings in every format, stream-set / data / backend preflights | Every number `compare` prints is labelled with what it is; structurally unfair comparisons refuse |
| 2     | Randomized PIT, fair CRPS, external score oracles, unified trace schema                                              | Score values match external oracles; calibration numbers are honest for count data                |
| 3     | `holdout_after` applied and scored (`--score-from` → `t0`), `Conditioning::HoldOutTail`, non-leakage preflight       | First honest out-of-sample number; "held out" is machine-verified, not asserted                   |
| 4     | Posterior-mixture predictive, replicate derives + Monte-Carlo SE, Newey–West SE                                      | Both optimism axes typed; Δ uncertainty includes filter noise and serial dependence               |
| 5     | Forecast-mode (no-assimilation) scoring, rolling-origin evaluation                                                   | Forecast skill measured across epidemic phases; single-tail atypicality resolved                  |

Named follow-ups, deliberately not staged here: LFO-PSIS (§3.9.1), h-block
interior windows (§3.9.2), SMC² (§3.9.3, long-term priority open), betting
display mode, chain stacking, the weighted interval score, ODE prequential
(gh#312), PIT-ECDF plots.

## 1. Background: what `compare` measures today

Facts verified against the tree at drafting time; the review pass re-verifies
each.

1. **The telescoping identity.** `particle_filter.rs` resamples unconditionally
   at every observation and resets log-weights to zero afterwards
   (`particle_filter.rs:539-591`), so the pre-observation weights are uniform
   and the recorded per-step log score `logsumexp(log_liks) − log N`
   (`prequential.rs:349`) equals the filter's likelihood increment
   (`particle_filter.rs:516`). Summing over steps,

   $$\mathrm{elpd}_{\text{preq}} \;=\; \sum_{t=1}^{T} \log \hat p(y_t \mid y_{1:t-1}, \hat\theta) \;=\; \log \hat p(y_{1:T} \mid \hat\theta),$$

   the plug-in conditional log-likelihood of the whole series. `compare`'s Δelpd
   is a likelihood-ratio comparison at two point estimates.

2. **The trace is honestly stamped** (`Provenance::PlugIn`,
   `Conditioning::InSample`, the `optimism_caveat` of gh#295 Ask 1) — but the
   caveat reaches stderr and the table renderer only, and its wording
   ("optimistic in absolute level") wrongly suggests differences are unaffected.

3. **`build_trace` accepts a `t0`** ("assimilated but not scored" — the filter
   reweights on observations before index `t0` but excludes them from the
   trace), tested by `build_trace_respects_t0_skip`, and both production callers
   pass `t0 = 0` (`pfilter.rs:780-781`, `fit/mod.rs:1939-1940`). This dormant
   parameter is the holdout scoring mechanism (§4.4, Stage 3). It is distinct
   from the conditioning window (`condition_from`), whose warm-up is _simulated
   but not scored_ — no assimilation (`fit/runner.rs:1476-1491`).

4. **Holdout declarations exist and are inert** (gh#585). `fit.toml` accepts
   `holdout_after` (temporal) and `[data.holdout]` (explicit files), validates
   them, and hashes holdout content into the fit's identity
   (`FitDigest.holdout_data`, gh#190) — and no code withholds or scores them.

5. **θ̂ for a Bayesian fit is the posterior mean over the (chain-filtered) draws
   cloud** (`compare.rs:719-767`); an optimizer fit uses its winner file.
   Derives are common-random-numbered across fits
   (`DEFAULT_DERIVE_PARTICLES = 1000`, `DEFAULT_DERIVE_SEED = 1`).

## 2. Decision record

Settled with the maintainer before drafting; recorded so this document is
self-contained.

1. `E_T` is relabelled now as the exponentiated log-likelihood ratio; e-value /
   betting vocabulary returns only for traces whose conditioning is honest
   (follow-up, not staged).
2. The Jeffreys evidence label is suppressed whenever `|Δelpd| < 2·se(Δ)`
   ("within noise"), with a standing note that the scale calibrates Bayes
   factors and these are not yet Bayes factors.
3. Newey–West SEs land in Stage 4; until then a small-`T` caveat accompanies
   `se(Δ)`. This is ordering by error budget, not a dependency.
4. The holdout arc (finish gh#585) is prioritized before LFO-PSIS.
5. The plug-in point stays the posterior mean until the posterior-mixture
   predictive (Stage 4) makes the choice moot.
6. Tail-only holdout in v1; interleaved holdouts are refused with an error
   naming the h-block follow-up (§3.9.2).
7. Holdout scoring defaults to assimilating one-step-ahead mode (§3.7.1);
   forecast mode (§3.7.2) is the Stage-5 variant.
8. Rolling-origin is fully specified here (§3.8) and staged last.
9. Mixture and replicate defaults: `M = 64` thinned draws, `R = 5` replicate
   seeds, both exposed as flags with a documented cheap mode.
10. SMC² is a named long-term follow-up whose priority is an open maintainer
    call; it gets its own proposal if promoted.

## 3. Methods

### 3.1 Notation and the prequential decomposition

Observations `y_{1:T}` at times `t_1 < … < t_T` (possibly multi-stream; the
joint score at a step sums the present streams, gh#636). Latent states `x_t`,
parameters `θ`. The prequential principle (Dawid 1984) scores the sequence of
one-step-ahead predictive distributions

$$p(y_t \mid y_{1:t-1}) \;=\; \int p(y_t \mid x_t)\, p(x_t \mid x_{t-1}, \theta)\, p(x_{t-1}, \theta \mid y_{1:t-1})\, dx\, d\theta.$$

Everything in this proposal is a choice of how the `θ`-marginal in that integral
is treated — which data `θ` is conditioned on (§3.2's conditioning axis) and
whether `θ` is a point or a distribution (§3.2's provenance axis). The state
marginal is always handled exactly (up to Monte Carlo) by the particle filter:
conditioning the _state_ on the past is what a forecast does, and is never a
leak.

For a state-space model, the bootstrap filter supplies the plug-in predictive at
zero extra cost as its per-step normalizing increment (pomp's `cond.logLik`;
King, Nguyen & Ionides 2016). That convenience is also the trap this proposal
exists to escape: with `θ̂` fit to the full series, the sum of those increments
is the in-sample likelihood (§1.1), and "prequential" in name does not make it
out-of-sample in fact.

### 3.2 The two optimism axes, typed

Following gh#295, a score's honesty decomposes into two orthogonal axes, carried
as typed metadata on every trace so no output can silently over-claim:

- **Provenance** — how `θ` enters: `PlugIn` (a single point; under-dispersed,
  because parameter uncertainty is dropped) vs `Posterior` (the predictive is
  mixed over the draws cloud, §3.6).
- **Conditioning** — which data `θ` has seen: `InSample` (all of it),
  `HoldOutTail` (a fit sealed before `τ`, §3.7), `Forecast` (no assimilation
  past the origin, §3.7.2), with `Lfo` reserved (§3.9.1).

The axes are independent: a posterior-mixture score over the full-data posterior
is honest in dispersion and still in-sample in conditioning. Both stamps exist
in the schema today; this proposal adds the missing variants and — critically —
makes conditioning a _preflight axis_ in `compare`: rows with different
conditioning modes are never differenced (§4.3).

The in-sample bias does not cancel in differences. For plug-in scores the
expected in-sample optimism is approximately the effective parameter count (the
argument underlying AIC), so Δelpd between models differing in flexibility
carries a systematic bias toward the more flexible one — a few nats for typical
model pairs, which is precisely the range the evidence scale labels
"substantial" to "strong." The Stage-1 caveat states this; Stages 3–5 remove it.

### 3.3 Scoring rules

**Log score.** `S(F, y) = log f(y)`; local, strictly proper (Gneiting & Raftery
2007). Estimated per step by the weighted mixture density over particles; with
uniform pre-observation weights this is `logsumexp(log p(y|x^{(i)})) − log N`
(verified identical to the kernel implementation).

**CRPS.** The continuous ranked probability score,
`CRPS(F, y) = ∫ (F(z) − 1{z ≥ y})² dz`, estimated from predictive samples. The
current estimator is the plain empirical form via the sorted-sample identity.
Stage 2 replaces it with the **fair** estimator (Ferro 2014; the pairwise form
below as written in Zamo & Naveau 2018), which is unbiased for the underlying
CRPS at finite ensemble size:

$$\widehat{\mathrm{CRPS}}_{\text{fair}} \;=\; \frac{1}{S}\sum_{s}|x_s - y| \;-\; \frac{1}{2S(S-1)}\sum_{s \ne s'}|x_s - x_{s'}|.$$

The difference from the plain `1/(2S²)` form is `O(1/S)` — negligible at
`S = 1000` held equal across fits by the derive settings, but it removes the
ensemble-size bias entirely, which matters when explicit `prequential.json`
inputs carry unequal `S`. Stage 2 also validates both estimators against the
`scoringRules` R package as an external oracle with committed reference cases
(gh#628), following the repo's external-oracle convention.

**PIT.** The probability integral transform `u_t = F_t(y_t)` is Uniform(0,1)
under calibration for continuous predictives (Gneiting, Balabdaoui & Raftery
2007). camdl's dominant likelihoods are counts, where the naive `P̂(X ≤ y)` PIT
is biased at atoms (gh#629) and coverage numbers inherit the bias. Stage 2
adopts the randomized PIT, in its sample form

$$u_t \;=\; \hat P(X < y_t) \;+\; v_t\,\hat P(X = y_t), \qquad v_t \sim \mathrm{Uniform}(0,1),$$

with the randomization seed recorded in the trace for reproducibility. Coverage
at nominal levels is then read off the randomized PIT exactly as today.

Attribution, load-bearing: the randomized transform is due to Smith (1985) and
Brockwell (2007), not to Czado, Gneiting & Held (2009) — CGH's own proposal is
the _nonrandomized_ PIT (a within-atom average rather than a draw), and they
cite the randomized form as prior work. We choose the randomized form because it
yields an exactly Uniform(0,1) value per observation that the existing per-step
`pit` field, coverage reads, and histogram consume unchanged; the nonrandomized
form is a mean of that distribution and would change those consumers' semantics.
CGH (2009) remains the citation for count-data predictive assessment; validation
cases (Stage 2.3) draw on its worked examples.

### 3.4 Uncertainty of Δelpd

Four distinct noise sources, each with its own remedy:

1. **Pointwise spread.** The paired SE `se(Δ) = √(T · Var_t(d_t))`,
   `d_t = ℓ^A_t − ℓ^B_t` (Vehtari, Gelman & Gabry 2017) — already implemented.
2. **Serial dependence.** Loss differentials are autocorrelated exactly when a
   model is misspecified — the interesting case — making the iid SE
   anti-conservative. The forecast-comparison standard is Diebold–Mariano (1995)
   with a heteroskedasticity-and-autocorrelation-consistent long-run variance
   (Newey & West 1987),

   $$\hat\sigma^2_{\text{NW}} \;=\; \hat\gamma_0 + 2\sum_{k=1}^{L}\Big(1 - \tfrac{k}{L+1}\Big)\hat\gamma_k,$$

   with the Harvey–Leybourne–Newbold (1997) small-sample correction and a
   `t_{T-1}` reference. Stage 4 lands this alongside the lag-1 autocorrelation
   of `d_t` as a printed diagnostic.
3. **Small `T`.** The normal approximation for Δelpd comparison is calibrated
   only when the scored count is large (order 100), the models are not too
   similar, and `|Δelpd|` is a few nats (Sivula, Magnusson, Matamoros & Vehtari,
   arXiv:2008.10296). Compartmental series routinely score 9–50 points. Stage 1
   attaches this precondition as a caveat whenever `T` is below threshold; no
   variance estimator rescues `T = 9`, so honesty in labelling is the fix.
4. **Filter Monte Carlo noise and bias.** The PF likelihood estimator is
   unbiased for `p̂` but negatively biased for `log p̂`, with bias equal to half
   the estimator's log-variance (the lognormal CLT of Bérard, Del Moral & Doucet
   2014), which grows with the weight variance of the specific model — so two
   models at the same particle count carry _different_ biases and part of Δelpd
   is differential Monte Carlo bias. Remedy (Stage 4): derive `R` replicate
   traces per fit at seeds `1..R`, combine with `logmeanexp` and its
   delta-method SE (already implemented in `evidence.rs::logmeanexp_with_se`,
   matching pomp practice), print the MC SE, and suppress the evidence verdict
   when `|Δ|` is within twice it.

### 3.5 Evidence vocabulary: what each scale licenses

Three quantities are habitually conflated; the output must not:

- **A likelihood ratio** `exp(Δelpd)` at plug-in points is what `compare`
  computes today. It licenses no betting or posterior-odds language.
- **A Bayes factor** is a ratio of _marginal_ likelihoods. The Jeffreys (1961)
  and Kass & Raftery (1995) scales calibrate Bayes factors. A prequential score
  becomes a log marginal likelihood only when the predictive at each step
  marginalizes `θ` over its _past-data_ posterior — the fully honest LFO/SMC²
  regime — at which point Δelpd is a log Bayes factor exactly and the scale
  applies.
- **An e-value / e-process** (Shafer 2021; Ramdas, Grünwald, Vovk & Shafer 2023)
  requires every forecast to be computable from the past alone; the Ville
  inequality's anytime-validity is void when `θ̂` has seen the future. Betting
  vocabulary is licensed only for honest-conditioning traces, and returns (with
  the archived proposal's disclaimer) only as a follow-up after Stage 3.

Stage 1 therefore: renames the `E_T` column `LR`, documents it as the
exponentiated in-sample log-likelihood ratio, strips e-value language from
`fmt_e_value` and its docs, gates the Jeffreys label on `2·se(Δ)` (decision 2),
and adds the difference-bias sentence to the optimism caveat.

### 3.6 The posterior-mixture predictive (provenance: `Posterior`)

For a Bayesian fit with post-warmup draws `θ_1..θ_M` (thinned; pooled across
retained chains through the existing `--exclude-chains` seam), the honest
predictive is the mixture

$$\hat p(y_t \mid y_{1:t-1}, \mathcal D) \;=\; \frac{1}{M}\sum_{m=1}^{M} \hat p(y_t \mid y_{1:t-1}, \theta_m),$$

computed by running one filter pass per draw and combining per-step _densities_
via `logsumexp_m(log p̂_{m,t}) − log M`. Averaging per-draw _log_ scores instead
is a different and strictly lower number (Jensen) that scores each `θ`
separately rather than the model's actual predictive — the distinction gh#295
records. For CRPS, PIT, and the fan-chart intervals, the predictive samples are
pooled across the `M` filters; the pooled cloud is a draw from the mixture, so
intervals widen to include parameter uncertainty — directly attacking the
plug-in overconfidence the `PIT_cov90` warning fires on. Cost: `M` filter passes
per fit, embarrassingly parallel, CAS-cacheable. Default `M = 64` (`--draws`),
with `--draws 1` documented as the cheap plug-in fallback (which then uses the
posterior mean, as today).

The mixture also supplies the scorer that chain stacking (Yao, Vehtari &
Gelman 2022) needs — score per-chain clouds separately, solve for weights — a
named follow-up.

### 3.7 Tail holdout (conditioning: `HoldOutTail`)

**Semantics.** A fit declares `holdout_after = τ` (or tail-only `[data.holdout]`
files). Training excludes all observations with `t > τ`; scoring runs the filter
over the _full_ series at the training fit's `θ` (point or mixture) and scores
only `t > τ`. Two scoring modes:

#### 3.7.1 Assimilating one-step mode (default)

The filter assimilates held-out observations as it scores them: each held-out
week is scored one-step-ahead, then enters the filtering state. Honest in `θ`
(the only leak channel), yields `T_hold` scored pairs, and is mechanically the
existing `build_trace` `t0` semantics — assimilated but not scored applies to
the _training_ window, scored starts at the first `t > τ`. Errors do not
compound; this mode measures one-step predictive skill on unseen data.

#### 3.7.2 Forecast mode (Stage 5)

No assimilation past `τ`: the ensemble propagates unconditioned through the tail
and each held-out week is scored against the `k`-step-ahead predictive,
`k = t − τ`. This is the deployed forecast product — what would have been
published at week `τ` — and it penalizes trajectory drift that §3.7.1 forgives.
Scores at different `k` come from one realized trajectory and are strongly
dependent: they are reported per horizon and never summed into one scalar.
Forecast mode is the machinery rolling-origin (§3.8) reuses.

#### 3.7.3 Machine-verified non-leakage

The claim "this score is out-of-sample" is verified, not asserted. A fit's
identity already seals: the canonicalized `fit.toml` (including
`holdout_after`), content digests of every resolved training stream, holdout
content digests (gh#190), and the engine version. `compare` refuses the
`HoldOutTail` stamp unless (a) the resolved fit's sealed config declares a
holdout window covering every scored time; (b) the fit carries the positive
proof that the window was actually applied — Stage 3.1 writes the applied
training window into `fit.meta.json`, and its _presence_ is the gate. A
version-comparison predicate ("engine post-dates Stage 3.1") is not computable
from the recorded `camdl_version` string — the git-hash suffix is unordered — so
the gate is the applied-window record, which a pre-Stage-3.1 fit (which parsed
`holdout_after` without applying it, and trained on everything) can never carry;
(c) the scored observations' stream/time set lies strictly after the training
window. The guarantee is precisely "no data leakage into `θ`": it cannot certify
that the _modeler_ chose the model family without seeing the full curve, and the
docs say so.

#### 3.7.4 What a tail score can and cannot conclude

The tail is where `R(t)` drifts, so tail scores are pessimistic relative to
mid-series interpolation — which is the deployment condition, not a bias. In
comparison the difficulty largely cancels (both models face the same tail),
residually favouring models that track regime drift. A short tail (6–10 points)
buys honesty, not power: the §3.4 small-`T` caveat applies with full force, and
single-split noise (the motivation of the original prequential proposal) is
answered by rolling-origin, not by a longer caveat.

### 3.8 Rolling-origin evaluation

Origins `τ_1 < … < τ_J` spanning the series (declared once in the toml, e.g. a
stride). For each origin: a fit on `y_{1:τ_j}` (each is simply a fit with
`holdout_after = τ_j` — CAS-cached, IF2 warm-startable), scored in forecast mode
over the following window; scores aggregated _per horizon_ across origins
(Tashman 2000; Hyndman & Athanasopoulos 2021, ch. 5). This is the standard
answer to single-split noise and phase-atypicality: the score averages over
rising limb, peak, and decline instead of being hostage to one tail. It aligns
camdl with epidemic-forecast-hub evaluation practice (Bracher, Ray, Gneiting &
Reich 2021; Held, Meyer & Bracher 2017), for which the weighted interval score
is a natural later addition (follow-up, not staged). Cost is `J` fits; `J` is
modest (an origin every 4 weeks) and each is cached.

### 3.9 Deferred methods (named follow-ups)

#### 3.9.1 LFO-PSIS

Approximate leave-future-out (Bürkner, Gabry & Vehtari 2020): fit once on an
initial window; approximate each later partial-data posterior `p(θ | y_{1:t-1})`
by Pareto-smoothed importance reweighting of the most recent fit's draws; refit
only when the tail-shape diagnostic `k̂ > 0.7`. Carried forward from the
superseded proposal with one camdl-specific risk now recorded: each draw's
incremental likelihood is itself a noisy PF estimate, and that pseudo-marginal
noise inflates the importance-weight variance, so `k̂` will trip more often than
in the Stan-model settings Bürkner et al. benchmarked — refit frequency, hence
cost, is an open empirical question. The Gelman–Vehtari book's preference for
h-block CV with joint log scores for the _comparison_ use (over LFO) is noted
and should be read before implementing.

#### 3.9.2 h-block interior windows

For comparison power: hold out an interior block, delete the `h` nearest
neighbours on each side from training (Burman, Chow & Nolan 1994; hv-block:
Racine 2000; validity under dependence: Bergmeir, Hyndman & Koo 2018), score the
block jointly, `θ` fit with the block holed out (the sparse-obs/NA machinery
already expresses this). Different estimand from a forecast score — `θ` has seen
the epidemic's future — so it gets its own conditioning label if implemented.
Deferred until rolling-origin's coverage is assessed.

#### 3.9.3 SMC²

Chopin, Jacob & Papaspiliopoulos (2013): a particle filter over `θ`-particles,
each carrying an inner state filter; the per-step normalizing constant is the
exactly-honest prequential predictive `∫ p̂(y_t|y_{1:t-1},θ) p(θ|y_{1:t-1}) dθ`
computed online, no refits, no importance approximation to diagnose. The
building blocks are the existing bootstrap PF (inner filter) and PMMH kernel
(rejuvenation). Long-term flagship; priority is an open maintainer call; if
promoted it receives its own proposal.

#### 3.9.4 Others

Betting display mode (`--show=betting`, gated on honest conditioning, with the
archived proposal's §6.4 disclaimer); chain stacking (Yao, Vehtari & Gelman
2022); the weighted interval score; ODE prequential (gh#312); PIT-ECDF plots
with simultaneous bands (the 2026-08-21 gap-analysis note, item B1).

### 3.10 References

- Bérard, J., Del Moral, P., & Doucet, A. (2014). A lognormal central limit
  theorem for particle approximations of normalizing constants. _Electronic
  Journal of Probability_, 19(94).
- Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of
  cross-validation for evaluating autoregressive time series prediction.
  _Computational Statistics & Data Analysis_, 120, 70–83.
- Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating
  epidemic forecasts in an interval format. _PLoS Computational Biology_, 17(2),
  e1008618.
- Burman, P., Chow, E., & Nolan, D. (1994). A cross-validatory method for
  dependent data. _Biometrika_, 81(2), 351–358.
- Brockwell, A. E. (2007). Universal residuals: a multivariate transformation.
  _Statistics & Probability Letters_, 77(14), 1473–1478.
- Bürkner, P.-C., Gabry, J., & Vehtari, A. (2020). Approximate leave-future-out
  cross-validation for Bayesian time series models. _Journal of Statistical
  Computation and Simulation_, 90(14), 2499–2523.
  doi:10.1080/00949655.2020.1783262.
- Chopin, N., Jacob, P. E., & Papaspiliopoulos, O. (2013). SMC²: an efficient
  algorithm for sequential analysis of state space models. _JRSS-B_, 75(3),
  397–426.
- Czado, C., Gneiting, T., & Held, L. (2009). Predictive model assessment for
  count data. _Biometrics_, 65(4), 1254–1261.
- Dawid, A. P. (1984). Statistical theory: the prequential approach. _JRSS-A_,
  147(2), 278–292.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
  _Journal of Business & Economic Statistics_, 13(3), 253–263.
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. _QJRMS_, 140(683),
  1917–1923.
- Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic
  forecasts, calibration and sharpness. _JRSS-B_, 69(2), 243–268.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. _JASA_, 102(477), 359–378.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
  prediction mean squared errors. _International Journal of Forecasting_, 13(2),
  281–291.
- Held, L., Meyer, S., & Bracher, J. (2017). Probabilistic forecasting in
  infectious disease epidemiology: the 13th Armitage lecture. _Statistics in
  Medicine_, 36(22), 3443–3460.
- Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and
  Practice_ (3rd ed.), ch. 5. OTexts.
- Jeffreys, H. (1961). _Theory of Probability_ (3rd ed.), Appendix B.
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. _JASA_, 90(430), 773–795.
- King, A. A., Nguyen, D., & Ionides, E. L. (2016). Statistical inference for
  partially observed Markov processes via the R package pomp. _Journal of
  Statistical Software_, 69(12).
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  _Econometrica_, 55(3), 703–708.
- Racine, J. (2000). Consistent cross-validatory model-selection for dependent
  data: hv-block cross-validation. _Journal of Econometrics_, 99(1), 39–61.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic
  statistics and safe anytime-valid inference. _Statistical Science_, 38(4).
- Shafer, G. (2021). Testing by betting. _JRSS-A_, 184(2), 407–431.
- Smith, J. Q. (1985). Diagnostic checks of non-standard time series models.
  _Journal of Forecasting_, 4(3), 283–291.
- Sivula, T., Magnusson, M., Matamoros, A. A., & Vehtari, A. Uncertainty in
  Bayesian leave-one-out cross-validation based model comparison.
  arXiv:2008.10296.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy. _IJF_,
  16(4), 437–450.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
  evaluation using leave-one-out cross-validation and WAIC. _Statistics and
  Computing_, 27, 1413–1432.
- Zamo, M., & Naveau, P. (2018). Estimation of the continuous ranked probability
  score with limited information and applications to ensemble weather forecasts.
  _Mathematical Geosciences_, 50(2), 209–234.
- Yao, Y., Vehtari, A., & Gelman, A. (2022). Stacking for non-mixing Bayesian
  computations: the curse and blessing of multimodal posteriors. _JMLR_, 23.

## 4. Implementation

### 4.1 Types

The trace schema moves to `schema_version: 3`; all additions are serde-defaulted
so v1/v2 traces still read.

```rust
// sim/src/inference/prequential.rs
// Externally tagged (serde's default) — deliberately NOT `tag = "kind"`:
// existing v2 traces wrote `"conditioning": "in_sample"` as a bare string
// (asserted by `conditioning_serializes_snake_case`), and internal tagging
// would fail to parse a present field of that shape. Externally tagged,
// `InSample` stays the bare string and the new struct variants serialize
// as `{"hold_out_tail": {...}}`, so v1 (absent field, serde default) and
// v2 (bare string) traces both keep reading.
#[serde(rename_all = "snake_case")]
pub enum Conditioning {
    /// θ from a fit that saw the scored window (today's only mode).
    #[default]
    InSample,
    /// θ from a fit sealed at train_end; scored steps all t > train_end;
    /// filter assimilates held-out observations as it scores (§3.7.1).
    HoldOutTail { train_end: f64, theta_source: String /* fit id */ },
    /// No assimilation past origin; per-horizon scores (§3.7.2, Stage 5).
    Forecast { origin: f64, theta_source: String },
    // Lfo { .. } reserved (§3.9.1).
}
// The String-carrying variants drop the current `Copy` derive on
// `Conditioning`; callers clone, as `Provenance` consumers already do.

pub enum Provenance {
    PlugIn,
    /// Mixture over M posterior draws (§3.6); records M and the draw seed.
    Posterior { n_draws: usize },
}
```

Additional trace fields: `pit_randomization_seed: Option<u64>` (Stage 2),
`score_from: Option<f64>` (the `t0` boundary as a time, Stage 3). The
`PrequentialTrace.t0` index field remains the mechanism; `score_from` is its
human-readable time-axis twin.

No `ir/VERSION` bump, no goldens: this touches `prequential.json` and CLI output
only. Behavioural re-keying is confined to fits that declare `holdout_after`
(§4.5).

### 4.2 Config and CLI surface

- `holdout_after = τ` (`fit.toml [data]`): now applied — training streams are
  truncated at load to `t ≤ τ`. Calendar-time models accept date strings through
  the shared time-spec grammar `condition_from` already uses
  (`fit/runner.rs::parse_time_spec`, the gh#626 grammar), which requires
  retyping the field from `Option<f64>` to the string-accepting spec form before
  parsing — same pattern `condition_from` follows.
- `[data.holdout]` explicit files: bound as the held-out tail; every held-out
  time must exceed that stream's maximum training time, else a hard error naming
  the h-block follow-up (decision 6).
- `camdl pfilter --score-from TIME`: maps `TIME` to the union-grid index and
  passes it as `build_trace`'s `t0`. Total loglik output is unchanged; the
  prequential trace scores only `t > TIME`.
- `camdl compare`: when every compared fit declares a holdout, derives held-out
  traces (full-series data = training ∪ holdout, `--score-from τ`) and stamps
  `HoldOutTail`; when none do, derives in-sample as today; a mix is a hard error
  (conditioning preflight, §4.3). `--draws M` and `--replicates R` control Stage
  4; `--in-sample` forces the old mode with the caveat.

### 4.3 Preflight matrix

All hard errors with explicit overrides, in the spirit of the existing `T_score`
check:

| Axis             | Check                                                             | Stage | Override                     |
| ---------------- | ----------------------------------------------------------------- | ----- | ---------------------------- |
| Horizon          | `T_score` equal (existing)                                        | —     | `--allow-mismatched-horizon` |
| Observation axis | per-step times equal (existing)                                   | —     | none (meaningless if bent)   |
| Stream set       | per-step per-stream name sets equal (gh#570)                      | 1     | none                         |
| Data             | bound-stream content digests equal across fits (gh#713)           | 1     | `--allow-data-mismatch`      |
| Backend/process  | derive refuses a backend the filter can't honestly score (gh#729) | 1     | none (until gh#312)          |
| Conditioning     | all rows same `Conditioning` kind                                 | 3     | none                         |
| Non-leakage      | §3.7.3 (a)–(c) for any `HoldOutTail` row                          | 3     | none                         |
| Provenance       | all rows same `Provenance` kind                                   | 4     | `--allow-mixed-provenance`   |

### 4.4 Stage-by-stage commit plan

One commit per numbered item; every fix lands red→green with a mutation check
(revert the source, keep the test, confirm red). Stage boundaries are the
reviewable breakpoints.

**Stage 1 — trust the table** (CLI layer only; no score values change):

1. Rename `E_T` → `LR`; rewrite `fmt_e_value` docs as exponentiated
   log-likelihood ratio; strip e-value vocabulary (`compare.rs:824-832`,
   headers, JSON key `e_t` → `lr`).
2. Gate the evidence cell on `2·se(Δ)`: below it, print "within noise"; add the
   Bayes-factor-scale footnote (`compare.rs:892-897`, `evidence.rs`).
3. Optimism caveat: add the difference-bias sentence (`prequential.rs:184-187`).
4. Emit warnings + provenance + conditioning + the optimism caveat in
   `--format json` and `md` (`compare.rs:975-1093`), per the archived proposal's
   §8.8 all-formats rule.
5. gh#570: stream-set preflight (extend `check_shared_observation_axis` to
   compare per-step `per_stream` name sets). Deliberate deviation from the
   issue's suggested override flag: no override — differencing joint scores over
   different observation sets is meaningless however displayed, same stance as
   the axis check. Guard the vacuous case: pre-gh#269 traces deserialize with
   empty `per_stream` (serde default); two empty-sided steps pass (nothing to
   check), empty-vs-nonempty at a paired step is refused as "one trace has no
   per-stream breakdown".
6. gh#713: data-digest preflight across the compared fits' bound streams, read
   from each fit's `fit.meta.json` sidecar (`FitSidecar.data_hashes`, stream
   name → sha256 — the readable artifact; `FitDigest` itself is hash-only and
   never written as JSON).
7. gh#729: refuse deriving a prequential for a fit whose backend the pfilter
   derive cannot honestly replay; read the stage's
   `backend() -> InferenceBackend` (`ChainBinomial | Ode`) from the resolved
   config — not `[synthetic].backend`, which only ever fed synthetic generation.
   Error names gh#312.
8. Small-`T` caveat line on `se(Δ)` (Sivula preconditions).
9. Whitepaper: correct the out-of-sample claim (`whitepaper.md:51`). Done during
   drafting (the paragraph now states the in-sample caveat); keep it in sync as
   stages land.
10. Correct the three shipped docs that describe holdout as working —
    `docs/fit-toml.md:56-57`, `docs/inference.md:1585-1597` (which shows a
    sample config and claims "Validate runs PF on train + holdout"), and
    `docs/camdl-inference-spec.md:67-71` — to state that the declarations are
    inert until Stage 3 (gh#585 calls the docs "the more urgent half"; leaving
    them standing through Stages 1–2 keeps inviting modelers to report in-sample
    numbers as held-out).

**Stage 2 — score integrity** (values change; oracle-pinned):

1. Randomized PIT (gh#629) with recorded seed; coverage reads unchanged.
2. Fair CRPS (Ferro 2014); update kernel tests.
3. External oracle: committed reference cases for CRPS (both estimators), log
   score, and PIT — `scoringRules` for the scores, CGH (2009)'s worked examples
   / `scoringutils` for PIT (gh#628; the PIT cases are what lets gh#629 close
   fully). R script writes inputs and answers, both committed.
4. Unify `prequential.tsv` schemas between `fit run` and `pfilter` (gh#650):
   hoist pfilter's `write_prequential_outputs` (currently private,
   `pfilter.rs:1419`) to a shared location and route the fit-stage writer
   (`fit/mod.rs:1972-1986`) through it.
5. ESS-collapse threshold becomes a fraction of `N` (currently absolute 10).
6. Warn when scoring starts at the prior (first scored step has no conditioning
   window and `t0 = 0`). While here: the dead
   `PrequentialWarning::
   UnderIdentifiedT0` variant (defined, never
   constructed) is either the carrier for this warning or gets deleted — not
   left dead.

**Stage 3 — holdout** (gh#585):

1. Apply `holdout_after` truncation and `[data.holdout]` tail binding at fit
   load; tail-only enforcement; the fit banner prints the training window, and
   the applied window is written into `fit.meta.json` (the §3.7.3(b) proof).
   Placement: the loading seam `resolve_and_load_obs_streams` is shared by
   `fit run`, `pfilter`, and `profile`, and compare's holdout scoring needs the
   full series — so the truncation goes at the fit-only callsite
   (`FitRunConfig::build`) or behind a parameter, never unconditionally inside
   the shared seam.
2. `pfilter --score-from TIME` → `build_trace` `t0`; trace records `score_from`.
   Wire both production call sites — `pfilter.rs:780-781` and the fit-stage
   trace at `fit/mod.rs:1939-1940` — not just pfilter's.
3. `Conditioning::HoldOutTail`; schema v3.
4. `compare` holdout auto-derive + conditioning preflight.
5. Non-leakage verification: declared window + the applied-window record in
   `fit.meta.json` (§3.7.3 a–c). A fit without the record — any fit produced
   before Stage 3.1 — is refused the `HoldOutTail` label.

**Stage 4 — honest uncertainty:**

1. Posterior-mixture predictive (`Provenance::Posterior`, `--draws`, logsumexp
   mixing, pooled samples), through the `--exclude-chains` seam.
2. Replicate derives (`--replicates`), `logmeanexp_with_se` combination, MC-SE
   column, verdict suppression within `2·`MC-SE.
3. Newey–West SE + HLN correction + lag-1 autocorrelation diagnostic.

**Stage 5 — forecast mode and rolling-origin:**

1. Forecast-mode scoring (no assimilation past origin; per-horizon rows; never
   summed across horizons).
2. Rolling-origin: origins spec extends the existing `compare.toml`
   (`CompareToml`, `deny_unknown_fields`) rather than a new file; each origin is
   a fit with `holdout_after = τ_j` (CAS-cached); per-horizon aggregation across
   origins.

### 4.5 Identity and re-keying consequences

- Fits whose configs set `holdout_after` change behaviour at Stage 3.1 (training
  truncates). Their identities re-key automatically — the engine version is in
  `FitDigest` — so no stale fit can be served for the new semantics. This is the
  alpha posture: no compatibility shim for the old (inert) behaviour, which was
  a bug (gh#585).
- `prequential.json` bumps `schema_version` 2 → 3 with serde-defaulted
  additions; old traces read as `InSample`/`PlugIn`, which is factually what
  they are.
- Mixture/replicate knobs (`--draws`, `--replicates`, `--score-from`) change
  stored derive output and belong in any CAS key that caches it
  (count-in-the-key rule). Note the blast radius: adding `score_from` to
  `PfilterConfigLevel` re-keys the _entire_ pfilter CAS namespace, including
  cached evals that never use the flag — correct under count-in-the-key, and
  broader than the fit-side re-keying above; cheap pre-1.0, stated so it is
  chosen, not discovered.

## 5. GitHub issue audit

Close candidates once the named stage lands (each closure confirmed by the
maintainer, per repo policy):

| Issue  | Covered by | Note                                                                                             |
| ------ | ---------- | ------------------------------------------------------------------------------------------------ |
| gh#570 | Stage 1.5  | stream-set preflight                                                                             |
| gh#713 | Stage 1.6  | data-digest preflight                                                                            |
| gh#729 | Stage 1.7  | refusal path; honest ODE scoring remains gh#312                                                  |
| gh#629 | Stage 2.1  | randomized PIT + the Stage 2.3 PIT reference cases (both halves of the ask)                      |
| gh#628 | Stage 2.3  | scoringRules oracle                                                                              |
| gh#650 | Stage 2.4  | unified trace schema                                                                             |
| gh#585 | Stage 3    | holdout applied, scored, verified; the docs half lands earlier, at Stage 1.10                    |
| gh#295 | Stages 1–4 | Ask 1 shipped earlier; close only after filing the LFO tracking issue that carries the remainder |

Advanced but not closed: gh#277 (held-out evaluation lands here; the emitter
ergonomics remain, and its `--holdout-strata` leave-district-out ask is actively
declined for v1 by decision 6 — it is h-block territory, §3.9.2), gh#312
(refusal only; ODE prequential is its own lift), gh#716 (compare reordering by a
sick chain — the fix is chain diagnostics, not compare), gh#633 (predict
draw-selection bias — adjacent, untouched).

Superseded document: `2026-05-31-prequential-bayesian-lfo.md` (its entire scope
is §3.9.1 plus items staged here); move to `docs/dev/proposals/archive/` when
this proposal ships.
