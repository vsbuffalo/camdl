# Diagnosing a fit that won't behave

A fit can fail in two fundamentally different ways, and they call for opposite
responses. Either the **model** is misspecified — it cannot generate data that
looks like what you observed, no matter the parameters — or the **inference** is
failing — the model is fine but the sampler or filter cannot navigate to the
answer. The symptom you see (chains that won't mix, a likelihood that won't
climb, an estimate pinned at a bound) usually does not tell you which one you
have. Tuning the sampler when the model is wrong wastes hours; fixing the model
when the sampler is the problem does nothing. The first job is therefore
**diagnosis, not tuning**.

This page is the decision tree. For how the particle filter, IF2, PGAS, and NUTS
actually work — and what each diagnostic column means — see
[`camdl docs inference`](inference.md); this page assumes that mechanics and
focuses on _which question to ask first_.

## 1. First question: is it the model or the inference?

Do not touch a sampler knob until you know. The single highest-return test is
**synthetic self-consistency**: simulate data from the model at a plausible
parameter vector θ, then re-fit (or re-filter) and see whether you recover θ.

```bash
# Generate synthetic data at a known θ
camdl simulate model.camdl --params theta.toml --obs synth.tsv --seed 1

# Re-fit with a fit.toml whose [data] points at synth.tsv, then compare the
# recovered estimate to theta.toml. (Data path lives in the fit.toml's
# [data] block — there is no --data flag on `camdl fit run`.)
camdl fit run fit_synth.toml --seed 2
```

Read the result this way:

- **Recovers θ on synthetic but fails on the real data → misspecification.** The
  inference machinery works; the model cannot reproduce the real data. Stop
  tuning and fix the model (a missing mechanism — seasonal forcing, a second
  introduction, reporting structure — is the usual culprit).
- **Fails even on synthetic → it's the inference.** The data came from exactly
  this model, so any failure to recover θ is the filter/sampler, not the model.
  Now sampler diagnostics are worth your time.

This test belongs **early**, before any sampler tuning — it flips the entire
diagnosis and costs one simulate-plus-fit.

**Its blind spot: a defect shared by the generator and the scorer.** The test
compares one code path against another, so it can only see a defect that one of
them has and the other does not. When the same wrong behaviour sits on both
sides, simulate-then-refit recovers θ cleanly from data that was never right.
Concretely (gh#681): a real-valued compartment referenced in an observation
evaluates as zero on **both** the forward emitter and the value-path scorer, so
a model whose observation adds an environmental reservoir generates data with
the reservoir dropped, scores it with the reservoir dropped, and returns θ to
three digits. Green test, wrong model.

The general rule: **a self-consistency test cannot see a defect its two halves
share; a cross-path comparison can.** Two paths that are supposed to compute the
same quantity — `mh` and `nuts` on one model and one dataset — disagree when one
of them is wrong, and that disagreement is the only signal a shared defect
leaves. Keep running the self-consistency test first; reach for the cross-path
check when it passes and you still do not believe the fit.

## 2. "Looks fittable" ≠ "is fittable"

A likelihood landscape that looks smooth and peaked can still be unfittable, for
two reasons:

1. **A central value per grid point hides the per-evaluation noise the sampler
   actually eats.** For a stochastic model the likelihood at a fixed θ is itself
   a noisy Monte-Carlo estimate; a landscape that reports only a summary smooths
   that away.
2. **A likelihood-only landscape sampled over bounds ignores the prior**, so a
   direction the prior would downweight still looks freely explorable.

In camdl, `camdl survey` is exactly this likelihood landscape — it draws points
by Latin-hypercube over the declared `[estimate]` bounds. Two things to know:

- It does **not** silently smooth the noise. By default it runs several particle
  replicates per point and reports `loglik_se` (the replicate standard error on
  the log scale) and `mean_ess` alongside `loglik`, summarises the across-point
  SE distribution in `summary.json`, and warns when too many points exceed the
  ~1.7-nat reliability bar (Doucet et al.). **Read `loglik_se` and `mean_ess`,
  not just the shape of `loglik`.** A peak built from points with large SE or
  collapsed ESS is an artifact.
- It is **prior-free by construction**. The landscape is the likelihood, not the
  posterior, so a flat direction a prior would tame still looks explorable. Heed
  the bound-clustering warning (top points pinning against a bound) — that is
  the landscape telling you a direction is unidentified by the data alone.

## 3. Two inference failure modes — opposite fixes

If §1 pointed at the inference, separate these two, because their fixes are
unrelated.

**(a) An under-resourced particle filter — PMMH, IF2 _and_ PGAS.** Every
particle method degrades when the filter has too few particles. The **lever is
the same in all of them (more particles)**, but the **symptom differs by
method**, and reading the wrong symptom sends you to the wrong fix.

- **PMMH and IF2 consume the _marginal_ likelihood**, so their symptom is
  _noise_: the filter's log-likelihood estimate is too variable, acceptance
  collapses, and IF2's particle ranking becomes arbitrary.
- **PGAS conditions on a sampled latent path instead of marginalizing it out, so
  its symptom is _mixing_, not noise.** The conditional sequential Monte Carlo
  (CSMC) step has to renew the reference trajectory; with too few particles it
  rarely does, the chain barely moves in latent space, and what you see is
  **poor R̂ and low `trajectory_renewal`** — never a noisy loglik. Read
  `trajectory_renewal` in `camdl fit summary` (the `LowTrajectoryRenewal` and
  `DegenerateAncestorSampling` diagnostics fire on it); gh#685 will add
  per-chain filter ESS next to it, which is the direct reading.

**If you are running PGAS, this is your bullet — do not read past it into (b).**
The fix for a starved CSMC is particles, and it looks nothing like the fix for
geometry.

For PMMH and IF2, first ask whether the noise is even fixable: scale particles
at one fixed θ and watch the loglik standard deviation.

```bash
camdl pfilter model.camdl --params theta.toml --data cases.tsv \
    --replicates 20 --particles 1000 --output ll_1k.tsv
# repeat at --particles 4000, 16000 and compare the reported loglik ± SD
```

If the SD falls like $1/\sqrt{N}$, more particles help. If it **plateaus**, no
particle count saves you — this happens when many observation streams are
observed at the same time (high _effective observation dimension_), where the
bootstrap filter is structurally inadequate. `camdl pfilter --pf-health`
measures this directly (the Snyder et al. 2008 $\exp(\tau^2/2)$ implied-N
estimate); see [`camdl docs inference`](inference.md) (the `--pf-health`
section). The fix there is a different method, not brute-force N.

For PGAS the equivalent probe is one re-run at roughly 4× the particles,
comparing R̂ and `trajectory_renewal`. Measured on a national Ebola PGAS fit —
same model, same data, same config, same 8,000 sweeps, **only `particles`
differs** — with per-parameter R̂:

| `particles` | chains | `trajectory_renewal` | `r_eff` | `tau` | `q_comm` | `gamma` | `rho` | `rho_lab` |
| ----------- | ------ | -------------------- | ------- | ----- | -------- | ------- | ----- | --------- |
| 1,200       | 6      | 0.591                | 1.33    | 2.52  | 2.64     | 1.81    | 1.18  | 1.19      |
| 4,800       | 7      | 0.707                | 1.02    | 1.46  | 1.32     | 1.37    | 1.03  | 1.03      |

Every parameter improved, acceptance and divergence counts became healthy, and
one further chain cleared the initialisation check (hence 6 → 7 from an
unchanged config). Nothing about the model changed. Before that re-run the team
had spent a day reading these R̂ values as (b) and was about to reparameterise.

**The ordering lesson, which is the generalisable part: establish that the
instrument is adequate before concluding anything about the model or its
geometry.** An under-resourced filter produces symptoms that mimic
non-identifiability — poor R̂ on exactly the parameters a weakly-identified
direction would spoil. Rule out (a) first; it costs one re-run.

**(b) Geometry (ridges, flat or stiff directions)** stalls gradient-based
PGAS-NUTS, which is a different problem with a different fix (reparameterize,
tighten priors, or add identifying data). Conclude (b) only once (a) is ruled
out.

**Crucially: PGAS-NUTS is immune to the PF marginal noise of (a).** It runs on
the _smooth complete-data conditional_ likelihood — it conditions on a sampled
latent trajectory rather than marginalizing it out with a noisy filter — so
"PMMH is dead on this problem" does **not** imply "PGAS is dead." If marginal
noise is killing PMMH/IF2, PGAS is often still viable. (See the
marginalize-vs-condition contrast in [`camdl docs inference`](inference.md).)

**Immune to the noise is not immune to the particle count.** PGAS still runs a
CSMC sweep, and that sweep still needs particles to renew the reference
trajectory — the mixing bullet in (a). "PGAS does not eat the marginal
likelihood" is a reason not to fear `loglik_se`; it is not a reason to run PGAS
at a particle count chosen for a cheap smoke test.

**(c) When the filter can't be rescued in place — scaffold with `ode + mh`, and
read the dying filter as a signal.** If the particle filter is dying (ESS
collapsing, particles going extinct) and you are already at a high particle
count or capped, stop adding particles and switch to the **`ode + mh`** backend.
It scores the _deterministic_ marginal likelihood directly — there is no
particle filter, so there is nothing to degenerate — and it explores fast,
giving a working posterior even where the stochastic filter is hopeless. Treat
it as a triage scaffold, not a compromise.

And read an _un-fixable_ dying filter as evidence about the model. If more
particles do not help **and** the filter is healthier on synthetic data than on
the real data (§1), the model often cannot produce data consistent with what you
observed — so the filter cannot keep particles alive near the observations. That
is **observation-model misspecification**, distinct from the structural
obs-dimension collapse in (a): the cure is to fix the observation process, not
the sampler. The productive sequence is to get a clean `ode + mh` fit first, let
its residuals, calibration (§5), and compensation signatures (§6) point at the
observation-model gap, fix the observation process there, and only then return
to the stochastic fit — which is often viable again once the model is
well-specified. This is the exact arc of a 14-stream spatial model whose
bootstrap PF was dead on the joint streams: `ode + mh` converged in minutes; its
single shared dispersion `k` pinned at the floor with overconfident intervals (a
calibration tell, §5/§6); a partially-pooled per-stratum dispersion fixed the
observation model; and the better-specified, non-centered model then ran under
PGAS.

### Choosing the backend by regime, up front

Scale selects the tool — you can often skip the failure-mode tree by matching
the backend to the regime before you start:

- **Large-population, near-deterministic systems → `ode` (`mh` / `nl-sbplx`).**
  Demographic noise is $O(1/\sqrt{N})$, so the deterministic skeleton ≈ the
  stochastic mean; and the over-constrained latent makes PF/PGAS choke — the
  same near-determinism that makes the ODE _accurate_ is what starves the
  filter.
- **Small-population / fade-out regimes → `chain_binomial` (`PMMH` / `PGAS`).**
  Stochasticity is real and the latent is not over-constrained.
- **Many simultaneous observation streams → not the bootstrap PF, at any scale**
  (the (a) collapse).

### Diagnosing and fixing a geometry stall (b)

When PGAS-NUTS stalls on geometry rather than noise, three moves, in order.

**First, know what R̂ can and cannot answer.** R̂ measures whether the chains
_agree_; it does not measure whether the data _determine_ a parameter. On a
genuinely flat direction, better mixing improves R̂ too — the chains explore the
flat direction more fully and agree on the same wide distribution — so a falling
R̂ is not evidence of identification, and a high R̂ is not proof of
non-identification. The instrument for identifiability is **prior-to-posterior
shrinkage** — posterior width divided by prior width — because unlike R̂ it does
not route through between-chain agreement at all.

**But shrinkage still has to be _estimated_, and that needs adequate ESS.** A
pooled 90% interval computed from badly-mixed chains describes where the chains
happened to sit, not the posterior: it inflates when mixing is poor and tightens
when mixing improves, exactly as R̂ does. Both statistics moved together in the
Ebola fit of §3(a) when the particle count rose — `tau`'s R̂ went 2.52 → 1.46 and
its 90% posterior width 0.294 → 0.204, from 79% to 55% of the prior's 90% width
of 0.371. At the per-chain effective sample sizes those runs delivered (order
5), neither number is trustworthy; 55% of prior is the best available reading of
how much the data constrain `tau`, and it is still not a reliable one. So: reach
for shrinkage rather than an R̂ contrast when the question is identifiability,
quote the ESS beside it, and re-read §3(a) before calling anything geometry.

**Name the bad direction with R̂.** If one parameter's R̂ won't converge but the
fit reproduces the data fine, suspect a _non-identified combination_, not a bad
parameter. Look at the posterior correlation matrix; if two parameters are ≈ ±1
correlated, compute R̂ on the **identified combination** (e.g. the product
$\rho \cdot D_{50}$) versus the **orthogonal** direction (the ratio): the
combination converges (R̂ ≈ 1), the sloppy direction does not. That decomposition
names the direction the chains disagree on, in one line. Confirm it with
shrinkage before reporting it as non-identifiability — an R̂ contrast on its own
can be nothing more than a contrast between a quantity the sampler happened to
mix on and one it did not. (A sloppy ridge in the sense of Gutenkunst et al.
2007; gh#263 proposes to automate this as a post-fit report.)

**Don't be fooled by warm-up.** R̂ ≈ 1.1 with a tail that _looks_ converged is
usually just too-short burn-in inflating the statistic. Recompute split-R̂
(Vehtari et al. 2021) after discarding the warm-up transient — 1.12 → ~1.00 from
exactly this is common. (gh#262's warm-up-aware R̂ surfaces it live.)

**Reparameterize the funnel.** The highest-yield fix for a hierarchical/pooled
parameter is the **non-centered** form. A centered
$k_p \sim \mathrm{LogNormal}(\mu, \tau)$ funnels — chains are lost and R̂ is
hopeless under gradient-free MH. Write it non-centered instead:
$k_{\text{raw}} \sim \mathrm{Normal}(0, 1)$ and
$k = \exp(\mu + \tau \cdot k_{\text{raw}})$ via a `let` (Betancourt & Girolami
2015). Bonus: because the hierarchy now lives in a `let` rather than a _declared
hierarchical prior_, the non-centered form also sidesteps PGAS's
hierarchical-prior gate (gh#175) — so it is what lets a pooled model run under
PGAS at all.

**Reseed a frozen sampler.** If PGAS-NUTS freezes from a cold start (step size
collapsing to ~$10^{-4}$), seed it at the `nl-sbplx` / `mh`-on-`ode` MLE: clean
latent trajectories let NUTS navigate locally where it stalled from a bad start.
(This is the IF2→PGAS warm-start in [`camdl docs inference`](inference.md),
extended to the deterministic backends.)

## 4. Pinning parameters helps geometry, not PF noise

Fixing or pinning parameters (`--fixed name=value`) reduces dimension and can
unstick the **geometry** problem (b) — fewer ridges to climb. It does
**nothing** for the under-resourced filter of (a): the ESS at a given θ does not
depend on how many parameters are free, and neither does whether CSMC renews the
reference trajectory. If your problem is the filter, pinning parameters will
feel like it should help and won't. Attack the problem you actually have.

## 5. Use the right "predicted value" for the diagnostic

Three different predictions answer three different questions; using the wrong
one hides the very misfit you're hunting.

- **Free-forward (unconditional posterior-predictive)** —
  `camdl fit predict --fit fit.toml --horizon free_forward`. Replays the fitted
  model from the start, never re-anchored to data, sampling `y_rep` per
  posterior draw. Exposes _generative_ misspecification: can the model, run
  forward on its own, produce data like the observations? A drifting transition
  model produces a band that blows up over time.
- **One-step-ahead** — `camdl fit predict --fit fit.toml --horizon one_step`.
  The honest short-horizon forecast `p(y_t | y_{1:t-1})`: re-conditions on the
  data at every step, so it stays tight **iff** the filter can track the data.
  The right tool for _timing_ questions (does the model anticipate each next
  observation?). Chain-binomial only. A model can pass the free-forward check
  (wide enough to contain the data) yet fail one-step (can't predict next week),
  and vice versa — which is why both are worth emitting and carry a typed
  `horizon` column so neither is read as the other.
- **Conditioned / smoothed path** — `camdl pfilter --save-paths`. This is pulled
  toward the data by construction and will track it even for a misspecified
  model. It **cheats** for the purpose of model-checking: a smoothed ribbon that
  hugs the data is not evidence the model is right.

`fit predict` writes both predictive horizons (stacked in
`predictive/<stream>.tsv`, keyed by the `horizon` column) and the observed
series (`observed/<stream>.tsv`) under the run directory, ready to join on
`(time, <dims>)` and plot — one facet per stratum. Read the predictive band as a
**quantile ribbon** (`q05…q95`), never a mean: averaging stochastic replicates
with jittered epidemic take-offs smears the peak later and lower than any single
run. The divergence between the free-forward ribbon and the smoothed ribbon _is_
the diagnostic — see the "unconditional vs smoothing" plot in
[`camdl docs inference`](inference.md).

Separately from _where_ a prediction sits, check whether its **intervals** are
trustworthy — the calibration question the three views above don't answer. Two
standard checks: the **PIT histogram** (the probability-integral transform of
each observation under its predictive distribution — U-shaped ⇒ intervals too
narrow / overconfident, domed ⇒ too wide; Gneiting et al. 2007) and
**nominal-vs-empirical coverage** (does the 90% interval actually contain ~90%
of held-out points?). This is exactly what separated the overconfident
shared-`k` fit (U-shaped PIT) from the well-calibrated hierarchical-`k` fit in
§3(c).

## 6. Read the MLE for "compensation" signatures

A point estimate can be a symptom rather than an answer. Watch for:

- a parameter **pinned at a bound** (survey's bound-clustering warning and the
  IF2 chain-agreement ranges surface this);
- an **unphysical** value — $R_0 < 1$ for an outbreak that visibly grew,
  overdispersion slammed to its maximum, a reporting rate at 0 or 1.

These are the optimizer contorting one parameter to absorb a structural misfit
elsewhere. When you see one, the estimate is telling you the model is wrong, not
giving you the answer — go back to §1.

A common specific case: a single negative-binomial dispersion `k` **pinned at
its floor (maximum overdispersion) across all strata**. The data are burstier
than one shared `k` can represent, and the fix is not a looser bound but more
structure — a partially-pooled (hierarchical) per-stratum dispersion, fit
non-centered (§3(b)). A floor-pinned shared `k` and the overconfident intervals
it produces (§5) are the paired tell that sent the 14-stream model in §3(c) to a
hierarchical observation model.

## 7. The meta-lesson: a fighting sampler is doing model-checking

A framework whose particle filter degenerates and whose sampler stalls on a
misspecified model is, in effect, performing model criticism for you. That
pushback is a **feature**, not a defect to tune away. When the sampler fights
you, suspect the model before the sampler — §1 is how you confirm it.

## camdl-specific gotchas

Concrete things that trip up real fits, verified against the current code:

- **PGAS (and PMMH) require a prior on every estimated parameter.** A parameter
  with no `~` in the model and no prior in the fit toml is a **hard error** that
  names the offending parameters and the three remedies — not a silent fallback.
  Declare priors via `~` in the model, in `[estimate.<param>].prior`, or opt
  into flat priors explicitly. (IF2 and the NLopt optimizers ignore priors.)
- **The ODE backend does MLE via `nl-sbplx` / `nl-bobyqa`, and Bayesian via
  `mh`** (Metropolis-Hastings on the deterministic marginal likelihood). PGAS
  and PMMH are **chain-binomial only** — they need stochastic process variance
  the ODE backend doesn't have, so asking for them on `ode` is a hard error that
  points you at `mh`. Run `camdl fit methods` for the current matrix.
- **The PGAS trace's loglik column is named `log_complete_data_ll`** — the
  complete-data conditional value (it conditions on the full sampled latent
  path: initial + transition + observation density), a large-negative number,
  **not** a marginal/PF likelihood. (PMMH and `mh` report the marginal in their
  `log_likelihood` column.) Don't compare PGAS's `log_complete_data_ll` to a
  `camdl pfilter` loglik — they differ by orders of magnitude.
- **The bootstrap particle filter degenerates with many simultaneous observation
  streams.** Per-stream likelihoods multiply into one weight, so high
  observation dimension collapses ESS and more particles only buy
  $\exp(\tau^2/2)$ headroom. Measure it with `camdl pfilter --pf-health` before
  scaling N (§3a).
