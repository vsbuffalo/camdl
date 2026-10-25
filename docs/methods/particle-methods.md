# Particle Methods in camdl: Bootstrap PF, IF2, CSMC-AS

**Scope:** the three distinct particle-method implementations in camdl, the
algorithm each one runs, and where each is plugged in. All three sit under the
Sequential Monte Carlo (SMC) umbrella but solve different problems: filtering,
parameter inference by iterated filtering, and conditional smoothing inside a
Gibbs sampler.

**Authoritative code:**

| algorithm                                      | entry point                                                                | callers                                                                                                                                |
| ---------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Bootstrap PF (vanilla filter)                  | `rust/crates/sim/src/inference/particle_filter.rs:87` (`bootstrap_filter`) | `cli/pfilter.rs`, `cli/fit/runner.rs:404` (`run_quick_pfilter`), `sim/inference/prequential.rs`, IF2 fallback path in `cli/if2.rs:399` |
| IF2 (parameter-augmented PF)                   | `rust/crates/sim/src/inference/if2.rs:191` (`run_if2`)                     | `cli/if2.rs` (standalone CLI), `cli/fit/runner.rs` (production chain runner under `fit run`)                                           |
| CSMC-AS (conditional SMC w/ ancestor sampling) | `rust/crates/sim/src/inference/pgas.rs:681` (`csmc_as`)                    | `pgas.rs:1412, 1655` inside `run_pgas`                                                                                                 |

**Shared infrastructure:**

- `rust/crates/sim/src/inference/resampling.rs:17` — `systematic_resample`. Used
  by `bootstrap_filter`, the inner loop of IF2, and `csmc_as`.
- `rust/crates/sim/src/inference/types.rs` — `ParticleState`, `SwarmState`, the
  trait surface every method consumes.
- `rust/crates/sim/src/inference/traits.rs` — `ProcessModel` (the propagation
  kernel `step_one`) and `ObservationModel` (the per-observation log-density).
  Every particle method in this document operates against these two traits.

This methods note covers what each algorithm computes, the equation each
implements, and when each is the right tool. For the IF2 cooling schedule
specifically, see [`cooling.md`](cooling.md). For NUTS (used by PGAS to update θ
between CSMC-AS sweeps) see TODO `nuts.md`.

---

## 1. The umbrella: filtering, smoothing, parameter inference

Sequential Monte Carlo is a family of algorithms that approximate a sequence of
distributions $\pi_t(x_{1:t})$ with a weighted set of samples ("particles")
$\{(w_t^{(i)}, x_{1:t}^{(i)})\}_{i=1}^N$, updated by a propagate-weight-resample
cycle as $t$ advances.

The algorithms below differ along three axes:

| axis           | values                                                             | what changes                                                  |
| -------------- | ------------------------------------------------------------------ | ------------------------------------------------------------- |
| **target**     | filter $p(x_t \mid y_{1:t})$ vs smoother $p(x_{1:T} \mid y_{1:T})$ | bootstrap PF targets the filter; CSMC-AS targets the smoother |
| **θ-handling** | fixed θ vs perturbed θ                                             | the bootstrap PF holds θ fixed; IF2 makes θ part of the state |

A useful first-principles reference that sets up this taxonomy is Doucet &
Johansen (2009), "A tutorial on particle filtering and smoothing: Fifteen years
later," in the _Handbook of Nonlinear Filtering_. Everything below is a
specialization of the framework in that chapter.

---

## 2. Bootstrap particle filter (vanilla)

**File:** `rust/crates/sim/src/inference/particle_filter.rs:87`
(`bootstrap_filter`).

**What it computes:** an unbiased estimator of the marginal likelihood
$p(y_{1:T} \mid \theta)$ and a weighted particle approximation of the filtering
distribution at each $t$.

### Algorithm (Gordon-Salmond-Smith 1993)

Given fixed parameters $\theta$, observations $y_{1:T}$, and $N$ particles:

1. **Initialize** ($t = 0$): draw $x_0^{(i)} \sim p(x_0 \mid \theta)$, set
   $\log w_0^{(i)} = -\log N$.
2. **For each observation time** $t = 1, \ldots, T$:
   - **Propagate**: $x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}, \theta)$ via
     `ProcessModel::step_one` (one or more substeps depending on $dt$ vs
     observation spacing).
   - **Weight**: $\log w_t^{(i)} = \log p(y_t \mid x_t^{(i)}, \theta)$ via
     `ObservationModel`.
   - **Estimate increment**:
     $\log \hat{p}(y_t \mid y_{1:t-1}) = \log\!\left(\frac{1}{N} \sum_i \exp \log w_t^{(i)}\right)$
     (numerically: log-sum-exp minus $\log N$).
   - **Resample**: if ESS $< N/2$, draw indices via `systematic_resample` and
     reset weights to $-\log N$.

Total log-likelihood:
$\log \hat{p}(y_{1:T} \mid \theta) = \sum_t \log \hat{p}(y_t \mid y_{1:t-1})$.

### Effective sample size (ESS)

$$
\text{ESS}_t = \frac{\left(\sum_i w_t^{(i)}\right)^2}{\sum_i (w_t^{(i)})^2}
$$

with the standard convention that the sum is over un-normalized weights. ESS
ranges in $[1, N]$; a low value means weights are concentrated on few particles.
The threshold for triggering resampling is $N/2$ in code
(`particle_filter.rs:265` neighbourhood). Citation: Liu & Chen (1995) JASA
90:567 introduced the ESS formulation; Kong, Liu & Wong (1994) JASA 89:278 set
up the sequential imputation framework.

### Unbiasedness — why this matters

The estimator $\hat{p}(y_{1:T} \mid \theta)$ produced by the bootstrap filter is
unbiased: $\mathbb{E}[\hat{p}(y_{1:T} \mid \theta)] = p(y_{1:T} \mid \theta)$.
Del Moral (1996, 2004) gave the full derivation; see Doucet & Johansen (2009)
§3.5 for the textbook version. **Unbiasedness is exactly the property PMMH
relies on** — substituting $\hat{p}$ for $p$ in the MH acceptance ratio still
targets the correct posterior.

### Worked example

TODO: pick a 5-observation toy SIR, $N=100$, walk one observation step showing
weight update, ESS, log-likelihood increment. Cross-reference
`tests/particle_filter.rs:132`.

### Citations

- Gordon, N., Salmond, D., & Smith, A. (1993). Novel approach to
  nonlinear/non-Gaussian Bayesian state estimation. _IEE Proceedings F_, 140(2),
  107–113. The original bootstrap filter.
- Liu, J., & Chen, R. (1995). Blind deconvolution via sequential imputations.
  _JASA_, 90(430), 567–576. ESS in PFs.
- Doucet, A., & Johansen, A. (2009). A tutorial on particle filtering and
  smoothing: Fifteen years later. In _The Oxford Handbook of Nonlinear
  Filtering_. Modern canonical reference.

---

## 3. IF2: iterated filtering with parameter perturbation

**File:** `rust/crates/sim/src/inference/if2.rs:191` (`run_if2`).

**What it computes:** a maximum-likelihood point estimate $\hat{\theta}$ of the
parameters by iterating a sequence of bootstrap-PF passes with the **state
augmented by per-particle parameter copies**, and the parameter copies perturbed
by Gaussian noise that cools toward zero.

### Algorithm (Ionides et al. 2015 PNAS)

State at time $t$ is augmented to $z_t^{(i)} = (x_t^{(i)}, \theta_t^{(i)})$. The
bootstrap-PF cycle is run with:

- **Perturbation step** (added to propagate): before each observation $t$,
  perturb
  $\theta_t^{(i)} \sim \mathcal{N}(\theta_{t-1}^{(i)}, \sigma_n^2 \Sigma_{\text{rw}})$
  where $\sigma_n$ is the cooling-controlled SD at iteration $n$.
- **Cooling**: $\sigma_n = \sigma_0 \cdot c^{2n/N_{\text{target}}}$ where $c$ is
  `cooling_fraction` (pomp's `cooling.fraction.50` convention). See
  [`cooling.md`](cooling.md) for the full treatment.
- **Same weighting and resampling** as the bootstrap PF: weights are still
  $\log p(y_t \mid x_t^{(i)}, \theta_t^{(i)})$; resampling draws joint
  $(x, \theta)$ pairs.

After $M$ iterations of this PF-with-perturbation, the parameter mean across
particles $\bar{\theta} = N^{-1}\sum_i \theta_T^{(i)}$ serves as the next
iteration's starting point. As $\sigma_n \to 0$, particles concentrate on the
MLE.

### Why this is not just a PF

The bootstrap PF assumes fixed θ. IF2 makes θ part of the propagated state and
lets the resampling step preferentially keep θ-copies that fit the data well.
The resulting algorithm is provably an MLE-targeting iteration (Ionides et al.
2015 give convergence under a Lyapunov argument). It is **not** posterior
sampling and the per-particle θ-clouds are **not** posterior draws — they
contract toward $\hat\theta$ by construction.

### Why camdl doesn't call `bootstrap_filter`

`if2.rs:191` runs its own particle loop because the propagation kernel must
perturb $\theta_t^{(i)}$ between substeps and the weight depends on the
per-particle θ. Reusing `bootstrap_filter` would require generalizing it to
"propagate joint $(x, \theta)$ state with per-particle θ" — possible, but the
IF2 loop has specialized cooling, parameter-bound enforcement, and dispersion
diagnostics that a generic interface would have to expose. Sharing just the
resampling step (`super::resampling::systematic_resample` at `if2.rs:461`) is
the pragmatic answer.

### Chain agreement Â

camdl reports a between-chain agreement statistic Â for IF2 chains. This is
**not** Gelman-Rubin's $\hat{R}$ — that is a posterior- sampling diagnostic; IF2
chains are MLE optimization trajectories, not posterior samples. The naming is
intentional: Â flags when optimizers have converged to the same neighbourhood,
and is part of the compound scout-convergence gate. See TODO
`chain-agreement.md`.

### Citations

- Ionides, E. L., Nguyen, D., Atchadé, Y., Stoev, S., & King, A. A. (2015).
  Inference for dynamic and latent variable models via iterated, perturbed Bayes
  maps. _PNAS_, 112(3), 719–724. **IF2.**
- Ionides, E. L., Bretó, C., & King, A. A. (2006). Inference for nonlinear
  dynamical systems. _PNAS_, 103(49), 18438–18443. The IF1 precursor; useful for
  understanding the iterated-filtering framing before perturbation was added.

---

## 4. CSMC-AS: conditional SMC with ancestor sampling

**File:** `rust/crates/sim/src/inference/pgas.rs:681` (`csmc_as`).

**What it computes:** a sample from the smoothing distribution
$p(x_{1:T} \mid y_{1:T}, \theta)$ conditional on retaining a prescribed
reference trajectory $x_{1:T}^*$ as one of the $N$ particle paths. The Markov
kernel this defines is **invariant under the smoother**, which is what makes
PGAS a valid Gibbs sampler.

### Algorithm (Lindsten-Jordan-Schön 2014)

Inputs: parameters $\theta$, observations $y_{1:T}$, reference trajectory
$x_{1:T}^*$, $N$ particles.

For $t = 1, \ldots, T$:

1. **Resample ancestors** for the $N - 1$ free particles. (The reference
   particle's slot is held; its ancestor is chosen by the ancestor-sampling step
   below, not the resampling step.)
2. **Propagate** the $N - 1$ free particles via
   $p(x_t \mid x_{t-1}^{(i)}, \theta)$. Set $x_t^{(N)} = x_t^*$ (the reference
   value at this time).
3. **Ancestor-sampling step (the AS in CSMC-AS)**: replace the reference
   particle's ancestor index $a_t^{(N)}$ by drawing from $$
   \Pr(a_t^{(N)} = j) \propto w_{t-1}^{(j)} \cdot p(x_t^* \mid x_{t-1}^{(j)}, \theta).
   $$
   This is what enables backward mixing without an explicit smoothing pass. See
   `pgas.rs:879` and surrounding for the buffered weight computation.
4. **Weight** all $N$ particles:
   $\log w_t^{(i)} = \log p(y_t \mid x_t^{(i)}, \theta)$.

After $T$ steps, sample one trajectory by drawing index $i \sim w_T$ and tracing
its ancestry; that trajectory is the new $x_{1:T}^*$ for the next Gibbs sweep.

### Why CSMC-AS, not just CSMC

Plain conditional SMC (Andrieu, Doucet & Holenstein 2010) is a valid Gibbs
kernel but mixes poorly when $T$ is large because the reference particle's
ancestry tends to dominate early-time slots (path degeneracy). Ancestor sampling
— Lindsten et al.'s contribution — replaces the reference's ancestor each
substep with a draw weighted by transition density, breaking the degeneracy. The
cost is one transition-density evaluation per particle per substep. See
`pgas.rs:443` (`complete_data_loglik`) and `pgas.rs:340`
(`log_transition_density_substep`) for the camdl implementation of that density.

### Diagnostics

`csmc_as` returns `CSMCDiagnostics` with `n_neg_inf_substeps` — the count of
substeps where ancestor weights all underflowed to $-\infty$. A nonzero count is
a red flag: it usually means the reference trajectory is incompatible with the
dynamics under the current θ (e.g., the chain is exploring a region where the
reference state at time $t$ is unreachable from any particle at $t-1$ within the
proposal noise). See `pgas.rs:148` for the diagnostic struct definition;
`docs/dev/incidents/2026-04-07-spatial-pgas-neg-inf.md` for a worked debugging
case.

### Worked example

TODO: 3-observation SEIR, $N=10$, reference trajectory provided, walk one
CSMC-AS sweep showing ancestor sampling at $t=2$. Cross-reference
`tests/spatial_density.rs`.

### Citations

- Lindsten, F., Jordan, M. I., & Schön, T. B. (2014). Particle Gibbs with
  ancestor sampling. _JMLR_, 15, 2145–2184. **CSMC-AS.**
- Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov chain Monte
  Carlo methods. _JRSS-B_, 72(3), 269–342. The PMMH/PG framework that CSMC-AS
  slots into.

---

## 5. When to use which

| if you want to...                                                | use                                                          | implemented as                                | typical caller                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------ |
| evaluate $\log p(y_{1:T} \mid \theta)$ at a fixed θ              | bootstrap PF                                                 | `bootstrap_filter`                            | `camdl pfilter`; loglik-eval / gate checks via `run_quick_pfilter` |
| find the MLE $\hat\theta$                                        | IF2                                                          | `run_if2` (its own loop)                      | `camdl fit run` with `[method] algorithm = "if2"`                  |
| sample from the posterior $p(\theta, x_{1:T} \mid y_{1:T})$      | Particle Gibbs with CSMC-AS for X-update + NUTS for θ-update | `csmc_as` inside `run_pgas`                   | `camdl fit run` with `[method] algorithm = "pgas"`                 |
| sample posterior with classical PMMH (Andrieu-Doucet-Holenstein) | PMMH on top of bootstrap PF                                  | `run_pmmh` calling `bootstrap_filter`         | `camdl fit run` with `[method] algorithm = "pmmh"`                 |
| score predictions sequentially                                   | bootstrap PF in prequential mode                             | `bootstrap_filter` with `PrequentialRecorded` | `camdl compare` via `prequential.rs`                               |

---

## 6. Cross-cutting choices documented elsewhere

- **Resampling.** All three methods systematic-resample by default. Tested in
  `sim/tests/particle_filter.rs::test_systematic_resample_*`.
- **Cooling schedule.** IF2-specific. See [`cooling.md`](cooling.md).
- **ESS thresholds and what they mean.** TODO `ess.md` (covers Liu-Chen 1995,
  the $N/2$ threshold convention, ESS-at-MLE reporting in loglik-eval).
- **Chain agreement Â** (between-chain MLE-optimizer convergence diagnostic —
  **not** Gelman-Rubin). TODO `chain-agreement.md`.
- **Compound scout-convergence gate** (Â + decibans-spread, SE-aware floor).
  Project-specific composition of standard pieces; needs its own methods note.
  TODO `scout-gate.md`.
- **Clean re-evaluation of IF2 final-iter mean.** Post-IF2, `coef(mif2_out)` is
  re-scored with `M` high-particle PF replicates and combined via `logmeanexp`
  on the likelihood scale — pomp's documented post-`mif2` workflow (King,
  Ionides & Bretó 2016 JSS; Ionides et al. 2015 PNAS supplement). The
  delta-method SE follows the same convention. **Not a separate methodological
  contribution** — at most a paragraph inside a future `if2.md`. See
  `docs/dev/notes/2026-04-27-clean-eval-strip.md` for the strip rationale (the
  prior 3-candidate construction was uncited and bias-on-top-of-bias-fix;
  stripped 2026-04-27).
- **NUTS** for θ-updates inside PGAS. TODO `nuts.md`. Hoffman & Gelman (2014)
  JMLR 15:1593.

---

## Authoritative citations (consolidated)

### Foundational SMC

- Doucet, A., de Freitas, N., & Gordon, N. (2001). _Sequential Monte Carlo
  Methods in Practice_. Springer. The canonical book.
- Doucet, A., & Johansen, A. (2009). A tutorial on particle filtering and
  smoothing: Fifteen years later. In _Handbook of Nonlinear Filtering_, Oxford.
  Modern reference; covers all three algorithms above as specializations of one
  framework.
- Gordon, N., Salmond, D., & Smith, A. (1993). Novel approach to
  nonlinear/non-Gaussian Bayesian state estimation. _IEE Proc-F_, 140(2),
  107–113. The original bootstrap filter.
- Liu, J., & Chen, R. (1995). Blind deconvolution via sequential imputations.
  _JASA_, 90(430), 567–576. ESS.
- Kong, A., Liu, J., & Wong, W. (1994). Sequential imputations and Bayesian
  missing data problems. _JASA_, 89(425), 278–288.

### IF2

- Ionides, E. L., Nguyen, D., Atchadé, Y., Stoev, S., & King, A. A. (2015).
  _PNAS_, 112(3), 719–724.
- Ionides, E. L., Bretó, C., & King, A. A. (2006). _PNAS_, 103(49), 18438–18443.
  (IF1 precursor.)

### Particle MCMC

- Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov chain Monte
  Carlo methods. _JRSS-B_, 72(3), 269–342. Introduces PMMH and Particle Gibbs.
- Lindsten, F., Jordan, M. I., & Schön, T. B. (2014). Particle Gibbs with
  ancestor sampling. _JMLR_, 15, 2145–2184. CSMC-AS.
