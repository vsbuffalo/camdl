# Inference in camdl

How the particle filter, IF2, PGAS, and NUTS work, what the diagnostics mean,
and how the inference pipeline fits together.

---

## The inference problem

A compartmental model defines a stochastic process over a **latent state** — the
compartment populations (S, E, I, R) evolving through time via stochastic
transitions. You never observe this state directly. What you observe is a noisy,
incomplete projection: weekly case reports, which are some fraction of
recoveries plus measurement noise.

The goal of inference is to estimate model parameters (transmission rate,
recovery rate, reporting probability, etc.) from the observed data. This
requires evaluating the **likelihood** — the probability of the observed data
given the parameters:

$$p(y_{1:T} \mid \theta) = \int p(y_{1:T} \mid x_{0:T}, \theta) \, p(x_{0:T} \mid \theta) \, dx_{0:T}$$

This integral is over all possible latent state trajectories $x_{0:T}$. For a
compartmental model with thousands of individuals tracked over hundreds of
weekly observations, this integral is intractable — you can't evaluate it
analytically.

But you _can_ simulate trajectories from $p(x_{0:T} \mid \theta)$. That's what
camdl does: given parameters, generate a stochastic realization of (S, E, I, R)
over time. The particle filter exploits this simulation ability to estimate the
intractable integral via Monte Carlo.

### Why this is hard

Three things make compartmental model inference harder than standard statistical
problems:

1. **Intractable likelihood.** The stochastic process (chain-binomial,
   Gillespie) doesn't have an analytic transition density. You can simulate from
   it but you can't evaluate $p(x_t \mid x_{t-1},
   \theta)$ in closed form. This rules out MCMC methods that require pointwise
   likelihood evaluation.

2. **High-dimensional latent state.** The state at each timepoint is the full
   vector of compartment populations. Over $T$ observations, the latent
   trajectory is $T$-dimensional. The likelihood integral is over this entire
   path space.

3. **Nonlinear dynamics.** Small parameter changes can produce qualitatively
   different behavior — biennial vs annual epidemic cycles, fadeout vs
   persistence, early vs late peak timing. The likelihood surface has ridges,
   local optima, and flat regions where different parameter combinations produce
   similar dynamics.

## How the algorithms relate

All four inference algorithms in camdl are built on Sequential Monte Carlo (SMC)
— the particle filter. They differ in what they do with the particles and what
they produce.

```
    Bootstrap Particle Filter
    (forward simulate, resample to match data)
                 │
                 │ used as a subroutine by:
                 │
    ┌────────────┼────────────┐
    │            │            │
   IF2         PMMH         PGAS
(find MLE)   (posterior)  (posterior + trajectories)
```

**IF2** (Iterated Filtering) perturbs parameters inside the particle filter and
cools toward the MLE. It's a stochastic optimization algorithm, not a sampler —
it finds the best-fit parameters but doesn't characterize uncertainty. Fast,
robust, good for finding the right basin.

**PMMH** (Particle Marginal Metropolis-Hastings) uses the particle filter's
log-likelihood estimate as the acceptance ratio in a Metropolis sampler. It
**marginalizes out trajectories** — the PF integrates over all possible latent
state paths, and PMMH only sees the marginal likelihood number
$\hat{p}(y|\theta)$. Any process model works (plug-and-play), but the PF
likelihood estimate is noisy, which slows mixing.

**PGAS** (Particle Gibbs with Ancestor Sampling) **conditions on a specific
trajectory**. It holds one complete latent trajectory $X$ fixed and evaluates
the exact complete-data likelihood $p(y, X | \theta)$ — no estimation noise.
Parameters are updated via NUTS or MH using this exact likelihood. The
trajectory is then refreshed via CSMC-AS (a particle filter conditioned on the
old trajectory). The Gibbs alternation ($\theta | X$ then $X | \theta$) samples
from the full joint posterior $p(\theta, X | y)$.

### Marginalizing vs conditioning on trajectories

This is the fundamental design choice:

|                   | PMMH                      | PGAS                                                                |
| ----------------- | ------------------------- | ------------------------------------------------------------------- |
| **Trajectories**  | Marginalized out by PF    | Conditioned on, explicitly sampled                                  |
| **Likelihood**    | Estimated (noisy)         | Exact (no PF variance)                                              |
| **Process model** | Any (plug-and-play)       | Chain-binomial only (needs transition density)                      |
| **Output**        | Posterior $p(\theta       | y)$ (trajectories available but low-quality due to path degeneracy) |
| **Bottleneck**    | PF variance → slow mixing | Trajectory convergence → slow on long series                        |

PMMH is more general (works with any simulator) but pays for it with noisy
likelihood estimates. PGAS is more efficient (exact likelihood) but requires the
ability to evaluate transition densities — currently only the chain-binomial
(Euler-multinomial) backend supports this.

### The recommended workflow

```
IF2 (scout → refine) → PGAS ([stages.pgas] init = "from_mle", init_mle = "refine")
```

IF2 finds the right basin quickly (global exploration via many particles). PGAS
characterizes the posterior within that basin (exact likelihood, NUTS gradient
proposals, posterior trajectory samples). Starting PGAS from IF2 results avoids
the trajectory convergence problem that plagues random starts.

## The particle filter

The particle filter (sequential Monte Carlo) estimates the likelihood by running
many parallel simulations and letting the data select which ones survive.

The key insight: instead of integrating over all possible state trajectories at
once, do it **sequentially** — one observation at a time. At each observation,
use importance sampling to focus computational effort on trajectories that are
consistent with the data seen so far.

### Particles are state trajectories

Each of the $N$ particles is an independent stochastic simulation of the full
compartmental model. At any time $t$, particle $i$ has its own state vector
$(S_i, E_i, I_i, R_i)_t$ — its own realization of the epidemic. The particles
all share the same parameters $\theta$ but differ in their random draws (which
individuals get infected, when they recover, etc.).

The ensemble of $N$ particles approximates the **filtering distribution**
$p(x_t \mid y_{1:t}, \theta)$ — the posterior over the latent state given all
data up to time $t$.

### Weights score particles against data

At each observation time $t$, each particle $i$ gets a **weight** proportional
to how well it predicts the observed data:

$$w_i^{(t)} = p(y_t \mid x_i^{(t)}, \theta)$$

This is the observation model likelihood — for example, the discretized Normal
probability of seeing 500 reported cases given that particle $i$'s projected
recoveries (scaled by reporting rate $\rho$) predicted 490.

If particle $i$ predicted well, $w_i$ is large. If it predicted poorly (e.g.,
projected 50 cases when 500 were observed), $w_i$ is tiny.

### Resampling focuses effort

After weighting, **bootstrap resampling** draws $N$ new particles from the
current $N$, with probability proportional to weights. Particles that predicted
well get duplicated. Particles that predicted poorly are discarded.

After resampling, all particles have equal weight, but they cluster around state
trajectories that are consistent with the data. The filter has used the
observation to update its belief about the latent state — this is Bayesian
updating via Monte Carlo.

### The likelihood estimate

The marginal likelihood of each observation is the mean weight:

$$\hat{p}(y_t \mid y_{1:t-1}, \theta) = \frac{1}{N} \sum_{i=1}^{N} w_i^{(t)}$$

The total log-likelihood is the sum over all observations:

$$\hat{\ell}(\theta) = \sum_{t=1}^{T} \log \hat{p}(y_t \mid y_{1:t-1}, \theta)$$

This estimate is **unbiased** (in expectation, it equals the true
log-likelihood). With more particles, the variance decreases. The estimate is
always a lower bound on the true log-likelihood — more particles can only
improve it.

### Effective sample size (ESS)

After weighting but before resampling, the weights are unequal. The **effective
sample size** measures how many particles are actually contributing useful
information:

$$\text{ESS}_t = \frac{\left(\sum_i w_i^{(t)}\right)^2}{\sum_i \left(w_i^{(t)}\right)^2}$$

- $\text{ESS} \approx N$: all weights are similar — every particle is useful.
  The observation is unsurprising given the model.
- $\text{ESS} \approx 1$: one particle has almost all the weight — the filter
  has **degenerated**. Only one trajectory out of $N$ is consistent with the
  data. The log-likelihood estimate is unreliable.

ESS is the primary diagnostic. It drops during epidemic peaks (where the data is
most informative and small differences in predicted incidence produce large
weight differences) and recovers during inter-epidemic troughs (where all
particles predict similar low incidence).

### One-step-ahead predictions

Before resampling, the weighted particle ensemble gives the **one-step-ahead
prediction**: what the filter expected to see at time $t$ before observing
$y_t$. The weighted mean and quantiles of $\rho \times \text{projected}_i$
across particles give prediction intervals.

If 90% of data falls within the 90% prediction interval, the model is
**well-calibrated** — its uncertainty is neither too wide nor too narrow.
Systematic prediction bias (always overshooting peaks, always undershooting
troughs) indicates model misspecification.

### What happens at each observation time

```
1. PROPAGATE: advance all N particles from t_{k-1} to t_k
   For each particle i, for each sub-step dt:
     - Evaluate propensities from particle i's state
     - Draw events (multinomial for chain-binomial)
     - Accumulate flows (infection counts, recovery counts, etc.)
   After 7 sub-steps (one week), each particle has its own state
   and its own incidence count since the last observation.

2. WEIGHT: score each particle against the data
   For each particle i:
     projected_i = cumulative recovery flow since last observation
     weight_i = P(observed_cases | rho × projected_i, observation_model)
   Particles that predicted close to the observed value get high weight.
   Particles that predicted far from it get near-zero weight.

3. AGGREGATE: compute the log-likelihood increment
   ll_k = log(mean(weights))
   This is the marginal probability of this observation given all the
   particles. Sum these over all observations to get the total loglik.

4. DIAGNOSE: ESS and prediction quantiles
   ESS = 1 / sum(normalized_weights²)
   When all particles agree, ESS ≈ N. When one particle dominates,
   ESS ≈ 1. Low ESS means the filter is degenerating — most particles
   are useless and the loglik estimate is unreliable.

   Prediction quantiles (q05, q50, q95) show what the filter
   expected BEFORE seeing the data. If the data consistently falls
   outside the 90% interval, the model is misspecified.

5. RESAMPLE: keep the good particles, kill the bad ones
   Systematic resampling: select particles proportional to their
   weights. A particle with 3× the average weight gets ~3 copies.
   A particle with near-zero weight gets killed.

   After resampling, all particles are equally weighted again.
   The diversity has decreased (some particles are copies) but the
   surviving particles are all consistent with the data so far.

6. RESET: clear flow accumulators for the next observation interval
```

### Incidence observations and the model origin

An incidence observation (`incidence(...)`, i.e. a `cumulative_flow` projection)
is scored against the flow accumulated over the window
`(previous observation, this observation]`. The very first window starts at the
model origin (internal time 0, or `t_start`), so an incidence row placed _at_
the origin has a zero-width accumulation window: its expected count is
identically 0. A positive count at the origin is therefore impossible (`-Inf`
likelihood), and `camdl pfilter` / `camdl fit` reject it before the filter runs
with a diagnostic naming the convention and the three remedies:

- drop the origin row;
- shift the observation times to interval _ends_ (date each row at the end of
  its accumulation window); or
- move the model origin earlier so the first observation has a full preceding
  interval.

A zero count at the origin is consistent with the zero-width window and is
accepted. (Prevalence observations — `current_pop` — read state at the instant
and are unaffected: there is no accumulation window.)

The opposite failure is an origin placed far _before_ the first datum — e.g. a
covariate-informed burn-in that starts dynamics years before the case data (the
third remedy above, overshot). Then the first window spans the whole pre-data
gap, and the first incidence count is scored against the flow accumulated over
that entire span — a wrong likelihood (gh#134). The fourth remedy covers this:
set `condition_from` (a top-level `fit.toml` key) to one cadence before the
first datum, so the pre-data span becomes an unscored warm-up and the first
observation is scored against one normal cadence:

```toml
condition_from = "first_obs - 1 week"
```

Conditioning is **explicit, not inferred** — `camdl fit` rejects an incidence
model with a wide pre-data gap and no `condition_from` (W329), naming the fix,
rather than guessing a boundary (which would fail silently on irregular data).
For a **multi-cadence** model (streams on different schedules) `condition_from`
is per-stream: a table with an optional all-streams `default` plus
per-observation-label **shadows** —

```toml
[condition_from]
default = "first_obs - 1 week"
es = "first_obs - 2 weeks" # shadows the `es` stream only
```

Resolution per stream: its shadow → else `default` → else none. See
`camdl docs fit-toml` (the `condition_from` section) and §3.9 of
`camdl-inference-spec.md`.

### Multi-cadence: streams on different schedules

Surveillance streams often arrive on different schedules — polio AFP (acute
flaccid paralysis) reported roughly monthly, environmental surveillance (ES,
poliovirus in sewage) sampled roughly biweekly. `camdl fit` / `pfilter` /
`profile` accept this directly: you supply one data file per stream (each on its
own time axis), and the filter merges them onto a **union observation axis** —
the sorted set of all streams' observation times.

Two things make this correct rather than a fudge:

- **Each stream is scored only at its own observation times.** At a union time
  where a stream has no observation (a sibling's reporting date), that stream
  contributes no likelihood term — it is simply not observed then.
- **Each incidence stream's bin closes on its own cadence.** The flow
  accumulator for an incidence stream resets only at _its_ observations, never
  at a sibling's. So an ES sampling date does not truncate AFP's monthly count;
  AFP's bin runs the full month regardless of how often ES reports in between.
  (Prevalence streams read state at the instant and never accumulate, so the
  reset doesn't apply to them.)

Each stream is conditioned independently (see `condition_from`, above — the
per-stream table form is exactly for this). A homogeneous model (every stream on
one cadence) is the special case where the union _is_ the shared axis and every
stream is observed at every time — and it fits byte-for-byte as before.

`camdl simulate --obs-dir <dir>` writes one file per stream at its own cadence,
which is the natural input back into a multi-cadence fit.

### The `--trace` output

```
time  ll_increment  ESS    pred_mean  pred_q05  pred_q50  pred_q95  observed
7     -7.84         17.4   42.3       5         31        112       82
14    -5.37         217.7  51.2       12        45        98        98
```

**ll_increment:** How surprising this observation was. More negative = more
surprising. A value of -3 means "this observation is about as likely as seeing a
specific card drawn from a deck of 20." A value of -10 means "this observation
is extremely unlikely given the model."

**ESS:** Effective sample size. Healthy range: 20-80% of N. Below 10% means the
filter is collapsing — increase N or check the model. Above 90% means the
observation is uninformative (the model already knew what to expect).

**pred_q05/q50/q95:** The filter's prediction before seeing the data. If
`observed` falls between q05 and q95 about 90% of the time, the model is
well-calibrated.

### CLI

```bash
camdl pfilter model.camdl --params p.toml --data cases.tsv \
    --particles 5000 --dt 1 --seed 42 \
    --flow recovery \
    --trace -
```

**`--flow recovery`**: Which transition's cumulative flow to use as the
projected quantity. Must match what the data measures.

**`--trace -`**: Write per-observation diagnostics (ll increment, ESS, the
filter's predictive quantiles vs the observed value) as a TSV. Use `-` for
stdout, or a path to write a file.

**Observation model**: not a CLI flag — it is declared in the model file's
`observations {}` block (e.g. `likelihood = neg_binomial(...)`). To change the
observation model, edit the model, not the command line.

**Likelihood floor**: when a particle predicts ~0 cases but the data shows 80,
both "predicted 0" and "predicted 5" are equally wrong — the filter floors the
per-particle likelihood so they are treated the same. Without the floor,
"predicted 0" gets a 650 log-unit worse penalty than "predicted 5", collapsing
ESS. The default matches pomp.

### Filtering marginals vs smoothing paths

At each observation step `t`, the bootstrap filter holds `N` particles weighted
by `p(y_t | x_t, θ)`. Two different distributions come out of this setup, and
conflating them produces quietly-wrong plots:

- **Filtering marginals** `p(x_t | y_{1..t}, θ)` — the per-step distribution of
  particle states at time `t`, weighted by their log-weights.
  `camdl pfilter --save-filtering PATH` dumps these as a long-format TSV.
  **Joining particles across `t` by index is NOT a sample path.** Resampling
  between steps shuffles the swarm; the particle indexed `i` at step `t+1` is
  not a descendant of particle `i` at step `t`.

- **Smoothing paths** — samples from `p(x_{1:T} | y_{1:T}, θ)`. Each path is a
  coherent latent trajectory consistent with all observations. Obtained via
  ancestor tracing: at the final step, sample a particle proportional to its
  weight, walk its ancestor chain backwards to collect the state at each earlier
  step. `camdl pfilter --save-paths N PATH` writes `N` such paths.

For **"does this fit match the data?" plots**, use `--save-paths`. Its quantile
ribbon over `N` paths estimates the smoothing marginal at each `t` — what the
model believes the latent trajectory was given all the data.

For **PF diagnostics** (particle degeneracy, ESS decay, obs-model sanity checks,
filter-implementation debugging), use `--save-filtering`. The per-step
log-weights are what you need to detect those pathologies; they're not what you
need to compare trajectories to data.

### The diagnostic plot: unconditional vs smoothing

A fitted stochastic compartmental model gives you three distinct views of the
data:

1. **Unconditional posterior predictive.** `camdl simulate --replicates
   N` at
   the MLE. "What does the fitted model predict a priori?"
2. **Smoothing over latent.** `camdl pfilter --save-paths N` at the MLE. "What
   does the model think the latent trajectory was, given the data?"
3. **Raw observations.**

Plot (1) and (2) as ribbons, (3) as points, side by side:

- If both ribbons track the data: well-specified model, inference worked.
- If (2) tracks the data but (1) misses it: **diagnostic of over- flexible
  process noise papering over structural mis-specification.** The PF
  log-likelihood is high because the model is flexible enough to thread through
  any data via stochastic fluctuations — not because it predicts well.
- If both miss the data: the fit is wrong.

The second case is pedagogically important and easy to misread. A reader seeing
(1) alone miss the data will conclude "the fit is bad"; a reader seeing (2)
alone track the data will conclude "the fit is good." Neither is right. The
divergence between them _is_ the diagnostic — teach it that way.

Background: `docs/dev/proposals/2026-04-19-pf-latent-trajectories.md`.

---

## IF2: turning the particle filter into an optimizer

IF2 (Iterated Filtering, Ionides et al. 2015) finds the maximum likelihood
estimate (MLE) — the parameter values that make the data most probable. It does
this without gradients, using only the ability to simulate forward.

### The key idea

In a regular particle filter, all particles share the same parameters. In IF2,
**each particle carries its own parameter vector.** Particle 1 might have
R₀=57.2, particle 2 might have R₀=55.8. Each simulates with its own R₀.

When the filter resamples, particles with good R₀ values survive and particles
with bad R₀ values die. The parameter cloud contracts around values that explain
the data. Add a cooling schedule that shrinks the perturbation over time, and
the cloud converges to a point — the MLE.

### What happens at each observation time (IF2 vs PF)

The structure is identical to the particle filter, with two additions:

```
1. PROPAGATE: same as PF, but each particle uses its OWN params
   particle_i simulates with particle_params[i], not shared θ

2. PERTURB: jitter each particle's parameters (NEW in IF2)
   For each particle i, for each estimated parameter:
     θ_i += Normal(0, rw_sd × cooling) on the transformed scale
   IVP parameters (initial conditions) are only perturbed at t=0.

3. WEIGHT: same as PF — score against data
4. RESAMPLE: states AND parameters are copied together (NEW in IF2)
   Good (state, θ) pairs survive. Bad pairs die.
5. RESET: same as PF
```

### The cooling schedule

The perturbation shrinks over time. After `cooling_target_iters` (50) full
iterations of the filter, the perturbation SD is `cooling_fraction` (0.95) of
the initial value.

```
Per-step cooling factor:
  c = 0.95 ^ (1 / (50 × n_observations))

After m iterations × n_obs steps each:
  effective_sd = rw_sd × c^(m × n_obs)
```

With 780 weekly observations, the cooling is very gentle per step (c ≈ 0.99993)
but compounds over many iterations. After 50 iterations: SD is 95% of initial.
After 100 iterations: ~90%. After 200: ~81%.

Early iterations: wide exploration (parameter cloud spans a broad range). Late
iterations: fine tuning (cloud contracts to a tight point).

### Parameter transforms and bounds

Parameters live on different scales. R₀=56.8 and ρ=0.488 need different
perturbation strategies.

| Parameter type          | Transform    | Why                             |
| ----------------------- | ------------ | ------------------------------- |
| `positive in [a, b]`    | Scaled logit | Bounds enforced by construction |
| `rate` (unbounded)      | Log          | Multiplicative perturbation     |
| `probability in [0, 1]` | Logit        | Stays in (0,1)                  |

The transform is derived automatically from the DSL parameter declaration. A
parameter declared `R0 : positive in [1, 100]` uses scaled logit — the
perturbation happens on (-∞, ∞) and the inverse transform maps back to [1, 100].
R₀ can never leave its bounds.

**rw_sd is on the natural scale.** `--rw-sd "R0=5"` means "perturb R₀ by about 5
units per step." Internally, this is converted to the transformed scale via the
delta method: for log-transformed params, the effective SD on log scale ≈ rw_sd
/ current_value. For R₀=56.8 with rw_sd=5, the perturbation is ~9% per step on
the natural scale.

**Scale warnings:** If rw_sd is >50% of the parameter value, the perturbation is
dangerously large. If <0.1%, the parameter isn't exploring. The CLI warns with
suggested adjustments.

### Multi-chain and chain-agreement Â

Run multiple independent IF2 chains from different random seeds to detect
multimodality and assess convergence. A single-method IF2 fit is a fit with one
`algorithm = "if2"` stage:

```toml
# fit.toml
[stages.fit]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.7
```

```bash
camdl fit run fit.toml --seed 42
```

**Chain-agreement Â** measures across-chain agreement (Gelman–Rubin 1992 form,
applied to IF2's per-iteration parameter-mean trajectory across chains; this is
**not** a posterior mixing statistic — IF2 is an MLE optimizer, not a sampler,
so Â here measures whether the optimizer's chains agreed on a basin, not whether
a posterior has mixed). Computed from the last half of iterations:

- Â < 1.1: converged (✓) — chains agree
- Â 1.1–1.5: uncertain (~) — might need more iterations
- Â > 1.5: not converged (✗) — surface may be multimodal

Note: Bayesian (PGAS, PMMH) outputs continue to use the name `rhat` for their
own posterior-mixing diagnostics; only the MLE pipeline (scout / refine /
validate) uses `chain_agreement` / Â.

### Regimes: scout → refine → validate

The typical MLE workflow is three `[stages.X] algorithm = "if2"` blocks in a
`fit.toml`, run in order by `camdl fit run`. Each stage warm-starts from an
earlier one via `init_mle = "<stage>"`. The three regimes differ only in their
stage knobs:

**Scout** — 8 chains, 500 particles, 30 iterations, **cooling = 0.70 (mild)**.
Exploration: chains stay hot enough to wander across basins rather than
quenching onto the first local optimum. Over the 30-iter stage the perturbation
SD shrinks only from 1.0× to 0.49× initial. Run this first to find problems: Is
the surface multimodal? Which parameters are identifiable? Is the observation
model appropriate? The cross-chain Â at the end of the scout, combined with the
loglik-eval decibans-spread gate (see camdl-inference-spec §6.1.1), is the
multi-modality diagnostic.

```toml
[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 500
iterations = 30
cooling = 0.70
```

**Refine** — 4 chains, 1000 particles, 50 iterations, **cooling = 0.05
(aggressive)**, `init_mle = "scout"`. Starts from the scout's best-chain
parameters and collapses chains tightly onto the local MLE — final SD is 0.25%
of initial, so particle clouds concentrate near the scout's endpoint. Check Â
for convergence across chains.

**Validate** — 4 chains, 5000 particles, 100 iterations, **cooling = 0.05**,
`init_mle = "refine"`. Full convergence for publication-quality estimates.

Cooling is pomp's `cooling.fraction.50` (cf50) convention: the parameter is the
halfway-point SD fraction, the end-of-stage SD is its square. Formula, worked
example, and empirical iter-by-iter table: `docs/methods/cooling.md`.

### IVP parameters

Initial conditions (S₀, E₀, I₀) set the starting state but don't change during
simulation, so they are perturbed only at t=0, not at every observation. They
are model-declared parameters: list them under `[estimate]` in the `fit.toml`
like any other parameter, and the fit estimates them. The filter jitters their
initial values once when particles initialize, then holds them fixed as it runs
forward — whereas a parameter like R₀ is perturbed at every observation time.
PGAS draws stochastic initial states from these parameters (e.g.
$S_0 \sim \text{Binomial}(N_0, s_0)$); see "IVP parameters (s0, e0)" below.

---

## Profile likelihoods

Fix a focal parameter at a grid of values, run IF2 at each to maximize over the
remaining parameters. The resulting curve shows how the MLE changes — revealing
identifiability, confidence intervals, and parameter correlations.

### 1D profile

```bash
camdl profile model.camdl --init from_params --params p.toml --data cases.tsv \
    --sweep "R0=lin(10,80,8)" \
    --rw-sd "sigma=0.01,gamma=0.01" \
    --particles 500 --iterations 30 --starts 3 --parallel 8
```

Output: TSV with R₀, max loglik at each grid point, and the estimated values of
all other parameters.

A sharp peak means R₀ is well-identified. A flat profile means R₀ is not
identifiable from the data (the model fits equally well across a range of R₀
values).

### 2D profile

```bash
camdl profile model.camdl --init from_params --params p.toml --data cases.tsv \
    --sweep "alpha=0.85,0.90,0.95,0.99" \
    --sweep "gamma=0.06,0.08,0.10,0.12" \
    --rw-sd "R0=2,sigma=0.01" \
    --particles 500 --starts 2 --parallel 8
```

Shows ridges and correlations between parameters. An elongated contour along the
alpha-gamma diagonal means those parameters trade off — you can't identify both
independently.

### Priors and precedence

Both `camdl fit run` and `camdl profile --algorithm pmmh` resolve priors for
each estimated parameter via a three-tier precedence chain: fit-toml > model-IR
`~` syntax > flat. The behaviour at tier 3 differs between the two subcommands
(warn vs error) — see below.

**Tier 1 — fit-toml priors (highest).** The fit toml's
`[estimate.<param>.prior]` block wins over any other source. For `fit run` the
fit toml is always loaded; for `profile` it's the `--fit <toml>` flag.

```toml
[estimate]
beta = { bounds = [0.01, 5.0], prior = { log_normal = { mu = -0.3, sigma = 0.5 } } }
```

**Tier 2 — model-IR `~` priors (fallback).** When the fit toml doesn't declare a
prior for an estimated parameter, the resolver falls through to whatever the
`.camdl` file declared via `~` syntax.

```
parameters {
  beta : rate in [0.001, 5.0] ~ log_normal(mu = -0.3, sigma = 0.5)
}
```

This is the recommended source of truth for stable priors: the model file is the
single artifact reviewers read for the _structural_ priors, and individual fit
tomls only override when doing sensitivity analysis. Stripping prior duplicates
out of N fit tomls and into one model file is what gh#75 unblocked.

**Tier 3 — `Prior::Flat` (last resort).** Improper uniform. The behaviour at
this tier is **asymmetric across subcommands**:

- `camdl profile`: tier 3 is a _silent fallback_ that emits a structured warning
  naming the affected parameters and citing the two remedies (declare in the
  model file or supply via `--fit`). Suppress with `--suppress-warnings` (loud —
  the waiver is recorded into `run.json`'s `suppressed_warnings`). Per-cell PMMH
  with flat priors is recoverable by spot-checking per-cell parameter values, so
  silent-with-warning is the right shape.

- `camdl fit run`: tier 3 is a **hard error** at config-load time (the fit
  refuses to start). The downstream interpretation of a fit-run chain —
  canonical posterior in `fit_summary.json`, consumed by tooling that treats
  those samples authoritatively — is too high-stakes for a silent demotion to
  "scaled likelihood". Users who genuinely want flat priors declare it
  explicitly via

  ```toml
  [estimate.beta]
  prior = { flat = {} }
  ```

  This opt-in path is fully accountable: the toml records the intent,
  `run.json`'s `resolved_priors` records the source as `flat_explicit`, and no
  warning fires (the user said what they meant). Silent fallback to flat is
  unreachable.

When all three tiers fail (no fit-toml prior, no model `~`, no explicit-flat
opt-in), `camdl fit run` errors with a 2-column table naming every offending
parameter plus all three remedies:

```
error: stage 'posterior' (method=pmmh) has parameters with no resolved prior:

  beta        no prior in fit toml, no `~` in model file
  gamma       no prior in fit toml, no `~` in model file

To proceed, do one of:

  (i)   Declare `prior = { <dist> = { ... } }` in the fit toml's
        [estimate.<param>] for each listed parameter.
  (ii)  Declare a `~ <dist>(...)` prior in the model file for
        each listed parameter.
  (iii) Opt into flat priors explicitly via
        `prior = { flat = {} }` in the fit toml — only do this if you
        intentionally want the chain to target the unconditioned
        likelihood (scaled-likelihood posterior).
```

```bash
# Profile-posterior sweep: --fit supplies priors and bounds
camdl profile model.camdl --data cases.tsv \
    --sweep "tau=lin(-35,-1,30)" \
    --algorithm pmmh --pmmh-steps 1500 --pmmh-particles 800 \
    --rw-sd auto --starts 3 \
    --fit fits/profile_tau.toml \
    --output results/profile_tau_posterior.tsv

# Bayesian fit with priors in the model file (no duplication in N fits)
camdl fit run fits/synth.toml --seed 0 --stage posterior
```

**Precedence rules** for parameter values (the unified chain shipped in the
2026-05-25 CLI UX revision; see
[`docs/camdl-run-spec.md §1.3`](camdl-run-spec.md) for the authoritative tier
list and
[`docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md`](dev/proposals/2026-05-25-cli-init-and-params-ux.md)
§"Precedence (last wins)" for the design rationale). Each tier overrides the
tier above it:

1. **Model parameter default.** The value declared in the `.camdl` source.
2. **`fit.toml` `[fixed]` block.** When `--fit` is in scope, every key in
   `[fixed]` overrides the model default.
3. **`--fixed-file <toml>`** (repeatable). A flat parameter TOML; top-level keys
   are parameter names. Layered in declaration order — later files override
   earlier ones.
4. **Scenario preset** (`--scenario NAME` or composed `--enable`/`--disable`).
   The active scenario's `params.set` / `params.scale` directives override
   everything above. _Scenarios travel with the model and beat user-supplied
   params files by design — choosing a scenario is choosing the model author's
   documented bundle._
5. **`--fixed NAME=VALUE`** (repeatable, highest). The per-invocation override;
   always wins. Use this for "I want to change one value for this one run."

The legacy `--params` and `--param` flags on inference subcommands (profile,
if2, fit run, survey) were removed in the same revision. Their replacements are:

- `--fixed-file <toml>` for the "load many values from a file" case.
- `--fixed NAME=VALUE` for the "change one value" case.
- `--init from_params --params <toml>` (a _companion_ of `--init`, not a
  top-level flag) for the _warm-start chain origin_ case — when the file is a
  starting point for inference, not a pin.

`--fixed`/`--fixed-file` on inference subcommands also removes the listed
parameter from the `[estimate]` set if it was there — so
`--fixed gamma=0.1 --sweep tau=lin(-35,-1,30)` is the canonical
slice-while-holding-gamma pattern. The kick-out is announced on stderr (one line
per parameter) so the override is never silent.

**Scenario-override visibility.** When a `--fixed-file` or `--fixed NAME=VALUE`
value overrides a value that the active scenario _also_ set, the resolver emits
a `ScenarioOverridden` warning to stderr at resolve time and records both values
into `run.json`'s `parameters_provenance` block:

```json
"beta": {
  "value": 0.5,
  "source": "fixed_cli",
  "role": "fixed",
  "overrode_scenario": {
    "scenario": "worst_case",
    "scenario_value": 0.3
  }
}
```

CLI override of a scenario value is a legitimate quick-test workflow; the
warning + provenance pair ensure six-months-later auditing of which value
actually ran isn't blocked by archaeology.

**Profile focal parameters.** The focal swept parameter(s) are always removed
from the estimated set, even when they appear in the fit toml's `[estimate]`.
Listing a focal parameter in `--fixed`/`--fixed-file` (or in the fit toml's
`[fixed]`) is a hard error — a parameter cannot be simultaneously swept and
pinned.

**Provenance.** Both subcommands write per-parameter `resolved_priors` into
`run.json`:

```json
"resolved_priors": [
  { "param": "beta",  "source": "fit_toml" },
  { "param": "gamma", "source": "model_ir" },
  { "param": "sigma", "source": "flat_explicit" }
]
```

The four wire values are `"fit_toml"`, `"model_ir"`, `"flat_explicit"` (gh#75 —
fit run only), and `"flat_fallback"` (profile only — the silent-fallback case).
Reviewers reading a fit dir's `run.json` can audit at a glance whether the chain
targeted a posterior with priors or the unconditioned likelihood.

The CAS hash includes the model IR bytes (`fit_content_hash` in `FitConfigV2`),
so re-running against the same fit toml after editing a `~` prior in the model
file produces a different cache dir. For profile, the CAS hash additionally keys
on `fit_toml_hash` + resolved per-parameter prior sources so re-running against
the same model with a different `--fit` flag produces a different umbrella.

### Per-cell diagnostics

Profile output gains a fixed-schema block of per-cell convergence columns
appended after the existing focal / loglik / parameter columns (gh#74 Option B).
The columns are written into the umbrella `summary.tsv` (the file `--output`
mirrors) and into each per-seed `profile.tsv`.

**Read by column name, not column index.** The diagnostic columns are the API;
their order is stable across runs, but defensive consumers should look them up
by name so future schema additions land cleanly.

The full list:

| Column                 | Algorithm | Meaning                                                                                                                                                                                                      |
| ---------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `acc_rate_avg`         | PMMH      | Mean MH acceptance rate across the K `--starts` chains. Post-burn-in (matches `PMMHResult.acceptance_rate`).                                                                                                 |
| `acc_rate_min`         | PMMH      | Minimum acceptance rate across the K chains. Surfaces "one chain stalled at 2%" failures the mean would hide.                                                                                                |
| `loglik_spread_starts` | all       | `max − min` of per-start final log-likelihoods. > ~5 nats means the starts disagree on the basin.                                                                                                            |
| `loglik_rhat_starts`   | PMMH, IF2 | Gelman–Rubin R̂ on the per-start log-likelihood traces (Gelman & Rubin 1992, _Statist. Sci._ 7(4); Brooks & Gelman 1998 corrected variant). > 1.05 is the conventional "chains haven't agreed yet" threshold. |
| `starts_n_completed`   | all       | Count of starts that produced a finite final loglik (didn't diverge).                                                                                                                                        |
| `iterations_used`      | IF2       | Final cooling-step index reached (= `--iterations` on normal completion).                                                                                                                                    |
| `cooling_final`        | IF2       | Mean across estimated parameters of the final iteration's `effective_rw_sd` — the _actual_ ending perturbation SD, not the target.                                                                           |

For algorithms that don't supply a given column (e.g. `acc_rate_avg` on an IF2
run, `iterations_used` on a PMMH run), the cell renders as `NaN` (capital N —
camdl's TSV NaN convention).

**The K<3 rule.** `loglik_rhat_starts` is `NaN` when fewer than three of the K
starts have a usable trace. Gelman–Rubin R̂ is undefined at K=1 and unstable at
K=2; the rule prevents a spurious diagnostic from a single-chain spike. To get a
finite R̂ supply `--starts 3` or more, _and_ run the per-cell inner loop long
enough to produce post-burn-in samples (`--pmmh-steps` must exceed the
hard-coded `burn_in = 100` for PMMH).

The diagnostic R̂ is computed on log-likelihood traces, not on per-parameter
posteriors, so it is a chain-level "are these starts walking the same basin"
check rather than a full multivariate convergence diagnostic. The column name
reflects that: it's `loglik_rhat_starts`, not `rhat`.

**Cross-seed aggregation.** For multi-seed profile runs (`--seeds
1,2,3`), each
cell's diagnostic columns are averaged across seeds in `summary.tsv` (with
`starts_n_completed` summing rather than averaging — it's a total count).
Per-seed values remain visible in each `replicates/seed_<n>/profile.tsv`.

---

## PGAS (Particle Gibbs with Ancestor Sampling)

IF2 finds the MLE. PGAS characterizes the full posterior — credible intervals,
parameter correlations, posterior trajectory samples.

### How it works

PGAS is a Gibbs sampler alternating two steps per sweep:

**Step 1: θ | X, y (parameter update).** With the full latent trajectory X
known, the complete-data log-likelihood is exact:

$$\log p(y, X \mid \theta) = \sum_s \log p(x_{s+1} \mid x_s, \theta) + \sum_t \log p(y_t \mid x_t, \theta)$$

No particle filter, no estimation noise. The transition density at each substep
is a product of Binomial log-PMFs mirroring the Euler-multinomial decomposition
in the simulation. Parameters are proposed via NUTS (gradient-based) or
one-at-a-time MH.

**Step 2: X | θ, y (trajectory update).** CSMC-AS (Conditional SMC with Ancestor
Sampling) produces a new trajectory sample from $p(X \mid \theta, y)$. One
particle slot is clamped to the reference trajectory; ancestor sampling at each
substep reconnects the reference to the free-particle cloud via the transition
density. Trajectory renewal (fraction of the traceback from non-reference
particles) measures CSMC health — near 0% means path degeneracy, above 50% means
healthy mixing.

### NUTS gradient proposals

The complete-data log-likelihood is differentiable with respect to parameters:
the Binomial log-PMF depends on rates via $p = 1 - \exp(-\text{rate} \cdot dt)$,
and the rates are differentiable expressions from the model.

The OCaml compiler performs source-to-source symbolic differentiation of rate
expressions (`autodiff.ml`), emitting `rate_grad` fields in the IR JSON. The
Rust backend evaluates these derivative expressions via the same `eval_expr`
interpreter — no runtime autodiff, no finite differences.

NUTS (No-U-Turn Sampler, Hoffman & Gelman 2014) uses these gradients to propose
all parameters jointly via Hamiltonian dynamics. A two-phase warmup adapts both
the step size (dual averaging) and the diagonal mass matrix (empirical variance
from burn-in). The mass matrix rescales each parameter by its posterior
variance, so NUTS takes appropriately-sized steps in every direction.

### Running PGAS

```bash
# From IF2 starting point: declare in fit.toml as
#   [stages.pgas] init = "from_mle", init_mle = "validate"
camdl fit run fit.toml --stage pgas

# From random starts (overdispersed initialization):
#   [stages.pgas] init = "lhs"  (or omit; lhs is the default)
camdl fit run fit.toml --stage pgas --seed 42

# Force MH-within-Gibbs instead of NUTS
camdl fit run fit.toml --stage pgas --no-nuts
```

Configuration in `fit.toml`:

```toml
[pgas]
chains = 4
sweeps = 10000
particles = 100
burn_in = 2000
thin = 5
n_trajectories = 200 # posterior trajectory samples per chain
```

Output per chain: `trace.tsv` (parameters + log-likelihood per sweep),
`trajectories/trajectory_NNNNNN.tsv` (posterior latent state draws).

### IVP parameters (s0, e0)

Parameters that determine the initial state (like the initial susceptible
fraction s0) require special treatment. The complete-data log-likelihood is
invariant to them because the trajectory's initial state is stored, not
recomputed.

PGAS handles IVPs by making the initial state stochastic: each CSMC particle
draws $S_0 \sim \text{Binomial}(N_0, s_0)$ independently, giving the CSMC
diverse initial states to select among. A Binomial density term is added to the
complete-data LL to constrain s0 via the MH ratio. IVP parameters are
auto-detected at startup.

### Spatial models and seeding (iota)

Spatial models with inter-patch coupling need care to ensure inference works
correctly. Two issues arise that don't affect single-patch models:

**Seeding terms.** If the infection rate for patch $i$ is
$\beta \cdot S_i \cdot I_i / N_i$, it goes to exactly zero when $I_i = 0$. The
stochastic simulator can still draw events from near-zero floating-point rates
(importation coupling creates tiny nonzero values), but the density evaluator
computes the rate as exactly zero and rejects the trajectory.

Fix: add a small seeding term to the infection rate:
$\beta \cdot S_i \cdot (I_i + \iota) / N_i$ where $\iota \approx 10^{-6}$. This
ensures the infection rate is never exactly zero, allowing importation-driven
infections to have finite (though very small) density. pomp spatial models use
the same pattern.

Not all spatial models need iota. Models with constant importation via
`events {}` blocks, or models where the rate expression already includes an
additive term, are fine without it.

#### Discrete-event seeding and PGAS

Discrete seeding via `events { ... add(E, n_seed) at [tau] }` is supported by
PGAS as of 2026-05-25 (gh#80). The chain-binomial simulator and the density
evaluator are consistent at the event substep: at substep $s_{\text{event}}$,
`counts_before` is the pre-event state, all stochastic flows are drawn from
pre-event rates (which are zero for the seeded compartment's downstream
transitions), and the event delta is applied atomically with transition deltas
at end-of-substep. The transition log-density at that substep is exactly zero
(finite). No Gaussian-pulse workaround is needed.

If a chain initialises with an event-time parameter (e.g., `tau`) outside the
simulation window, the seed never fires within the simulation, predicted
incidence stays zero, and the _observation_ density goes to $-\infty$ against
real data with nonzero cases. camdl prints a split-by-component diagnostic at
startup so the failure mode is identifiable (transition_ll vs observation_ll vs
ivp_ll); the chain will still run if NUTS or MH can propose into a feasible
region.

During CSMC ancestor sampling on event-using models, the density evaluator
returns $-\infty$ for a free particle whose pre-step state has zero rate for a
transition that fired in the reference's flow record. This is correct: the free
particle cannot be the ancestor and is excluded from the categorical.

**Time step size.** The Euler-multinomial approximation assumes exit
probabilities are small per substep. In spatial models with high $R_0$ and
$dt = 1$, $p_{\text{total}}$ can approach 1, causing overdrafts where total
withdrawals from a compartment exceed its population (resolved by clamping). Use
a smaller dt (e.g., 0.25) to keep $p_{\text{total}} < 0.3$ and avoid
approximation breakdown.

### MCMC initialization strategy

PGAS chains should be initialized at or near a known high-likelihood region, not
from random or diffuse starting points. The recommended workflow:

1. **IF2 scout:** Run 8–16 chains with random starts to map the likelihood
   basins. More chains are needed for spatial models where the surface is
   multimodal (R0–sigma–amplitude ridges create multiple basins).
2. **Profile likelihood:** Run a 1D profile over R0 (the parameter most prone to
   basin structure) to confirm which basin has the highest likelihood.
3. **Initialize PGAS:** Start all chains at the best IF2 MLE ± small jitter
   (e.g., ±5% per parameter). This avoids wasting burn-in searching for a basin
   that IF2 already found.

Starting chains near the mode is standard MCMC practice (Gelman et al., BDA3;
Stan's default workflow optimizes first, then samples). MCMC convergence
guarantees are asymptotic — initialization affects only burn-in length, not the
target distribution. Starting from a good point reduces wasted computation; it
does not bias the posterior.

**When initialization matters most:** Spatial models with seasonal forcing. The
R0–sigma trade-off creates basins separated by 50+ log-likelihood units. IF2
with only 4 chains can land in the wrong basin (e.g., R0≈28 instead of the true
R0≈20), and PGAS initialized there may never cross the barrier. More IF2 scout
chains is the fix — tempering can't bridge 50+ nat gaps either.

---

## Diagnostic interpretation guide

### Healthy pfilter trace

```
time  ll_increment  ESS    pred_mean  pred_q05  pred_q95  observed
7     -4.2          2800   45         12        95        52      ← data in interval
14    -3.8          3100   120        48        220       135     ← data in interval
```

ESS stays above 50% of N. Data falls within prediction interval. Log-likelihood
increments are moderate (not extreme).

### Degenerating filter

```
time  ll_increment  ESS    pred_mean  pred_q05  pred_q95  observed
7     -4.2          2800   45         12        95        52
14    -12.8         23     120        105       140       350     ← data far outside
```

ESS crashes to <1% of N. The data is very surprising given the model's
predictions. Causes: wrong parameters, wrong observation model, missing model
features (e.g., no seasonal forcing when the data has seasonal epidemics).

### IF2 convergence trace

```
iteration  loglik   R0      gamma
0          -6200    42.3    0.15     ← exploring
5          -4100    51.2    0.09     ← approaching
15         -3850    55.8    0.084    ← converging
30         -3810    56.5    0.083    ← stabilizing
50         -3805    56.8    0.083    ← converged
```

Log-likelihood should improve monotonically (with noise). Parameters should
approach stable values. If loglik oscillates without improving, rw_sd is too
large. If parameters haven't moved after 20 iterations, rw_sd is too small.

### IF2 chain-agreement diagnostics

```
Â (across 4 chains, last 25 iterations):
  R0           Â=1.02 ✓ range=[55.2, 58.1]
  sigma        Â=1.01 ✓ range=[0.078, 0.080]
  gamma        Â=3.20 ✗ range=[0.065, 0.120]
```

R₀ and sigma have converged (Â < 1.1, tight range). Gamma has not (Â=3.2, wide
range). This means gamma is either poorly identified or the surface is
multimodal along the gamma axis. Run a profile likelihood for gamma to
distinguish.

---

## The fit workflow

The low-level commands (`camdl pfilter`, `camdl profile`) are building blocks.
IF2 is not a standalone command — a single-method IF2 fit is just a fit with one
`algorithm = "if2"` stage. For all model fitting, `camdl fit` provides a
structured workflow driven by a `fit.toml` configuration file:

```
fit.toml + model.camdl + data.tsv
    │
    └── camdl fit run fit.toml
            <fit_dir>/real/fit_<seed>/
              ├── scout/    fit_state.toml      (stage, init = "lhs")
              ├── refine/   mle_params.toml     (stage, init = "from_mle", init_mle = "scout")
              ├── validate/ mle_params.toml     (stage, init = "from_mle", init_mle = "refine")
              └── pgas/     chain_N/trace.tsv   (stage, init = "from_mle", init_mle = "refine")
```

> **v2 layout note.** Stage directories live under
> `<fit_dir>/real/fit_<seed>/<stage>/` (or
> `<fit_dir>/synthetic/ds_NN/fit_<seed>/<stage>/` for SBC replicates). The
> `real/fit_<seed>/` and `synthetic/...` wrappers were introduced in commit
> `5f1e704` (2026-04-18) to support start-sensitivity and synthetic-data
> replicate grids; pre-2026-04-18 diagrams that show stages directly under
> `<fit_dir>/` are stale.

Each named block under `[stages.NAME]` in `fit.toml` chains via the
`init = "from_mle"` + `init_mle = "<prior-stage>"` pair. The default set is
scout → refine → validate (+ pgas), but users can define any sequence.

**Scout** (8 chains, 200 particles, no cooling): random starts across the
parameter space, MAD-based auto-calibration of rw_sd. Identifies the likelihood
basin and filters out divergent chains.

**Refine** (4 chains, 1000 particles, cooling=0.95): convergent IF2 from scout's
best parameters and auto-calibrated rw_sd. Produces an initial MLE.

**Validate** (4 chains, 5000 particles, cooling=0.95): final IF2 + profile
likelihoods for all estimated parameters + precise pfilter at the MLE for
log-likelihood and ESS measurement.

Each stage reads the previous stage's `fit_state.toml` and writes its own. The
final output is `mle_params.toml` — a standard params file with provenance
hashing that feeds directly into `camdl simulate` and `camdl batch run`.

```bash
# Full pipeline (all stages declared in fit.toml run in order)
camdl fit run    fit.toml --seed 1

# Re-run a single stage from a prior stage's output
#   (configured in fit.toml as `[stages.refine] init = "from_mle",
#    init_mle = "fit/he2010/real/fit_1/scout/"`)
camdl fit run    fit.toml --stage refine
camdl fit run    fit.toml --stage validate
camdl fit summary results/fits/<dir>/
```

### When `start =` is omitted in `[estimate]`

Each `[estimate.X]` entry's `start =` field is optional. When you omit it (and
the model file doesn't already declare a value for the parameter via
`parameters { X : rate = 0.3 }` or a scenario), the runner draws a single
Transform-aware uniform value within the parameter's bounds and uses that as the
base point. From there the selected `init` mode perturbs per-chain as usual.

The draw is:

- **Log-uniform** for `Log`-typed parameters with strictly positive bounds
  (rates, positive quantities) — equivalent to drawing uniformly in
  `[ln(lo), ln(hi)]` and exponentiating, so a parameter with bounds
  `[1e-6, 1.0]` doesn't collapse to a value near `0.5`.
- **Linear-uniform** for `Logit`/`None` parameters and for any parameter whose
  bounds aren't strictly positive.

The draw is deterministic per `(seed, parameter_name)`: re-running with the same
`--seed` gives the same fallback values, and two parameters with identical
bounds at the same seed get _different_ values (their names hash differently).
Different seeds give different fallback points within bounds — useful when you
want a seed sweep to also sweep over starting positions for the unspecified
parameters.

This replaces an earlier bounds-midpoint heuristic that gave the same point at
every seed and ignored the parameter's transform.

### Per-chain init: `init`

How chain (or per-cell) starting points are drawn. Set on each stage in
`fit.toml` via the `init = "<mode>"` key (or override per-stage on the CLI with
`--init`); also available as `--init` on `camdl profile` for per-cell starts.
Honoured by **IF2**, **PGAS**, **PMMH**, **NLopt** (`nl_sbplx`, `nl_bobyqa`),
and **profile**.

```toml
[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 16
init = "lhs" # this is now the default; shown for clarity
```

| Mode            | Behaviour                                                                                                                                                                                                                                                             | When to use                                                                                                                                                                                  |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lhs` (default) | Latin-hypercube stratified sampling, **scale-aware via the parameter's transform**: `Log`-typed rates are sampled in log space and exponentiated, so a single LHS pass spans orders of magnitude. `Logit`/`None`-typed parameters are sampled linearly in `[lo, hi]`. | The default for every multi-chain stage. Stratified coverage at the chain counts we typically run; supersedes the legacy `uniform` default.                                                  |
| `uniform`       | Per-chain uniform random within natural-scale bounds. Chain 0 keeps the seeded start.                                                                                                                                                                                 | Legacy mode. Equivalent to LHS for `Logit`/`None` parameters, but worse for `Log`-typed parameters at low chain count (clumps in linear space). Kept for reproducibility of pre-LHS results. |
| `single`        | Every chain at the seeded `[estimate].start` (or its `Transform`-aware uniform fallback when `start` is omitted). Chains differ only by per-chain RNG.                                                                                                                | See "When `single` is the right choice" below.                                                                                                                                               |

When a stage uses `init = "from_mle"` + `init_mle = "<prior>"`, every chain
starts from the prior stage's MLE — that's the intent of the handoff.

**Why LHS is the default.** With single-point starts (or clumpy uniform starts
at low chain counts), chains find one basin and miss the rest. On stratified epi
models with multiple modes, LHS-drawn starts reach basins that single-point
starts never see — on the typhoid stratified scout, 30 LHS chains beat 8
uniform-random chains by ~80,000 nats, holding everything else equal.

**When `single` is the right choice.** Four legitimate cases:

1. **Refine stages with `init = "from_mle"` + `init_mle = "<prior>"`** — all
   chains start from the prior stage's MLE anyway; `single` is redundant but
   harmless.
2. **Single-chain runs (`chains = 1`)** — there's no per-chain spread to draw,
   so the three modes collapse to the same draw.
3. **Reproducibility-critical tests** — `single` gives byte-identical chain
   starts across runs at the same seed; LHS/uniform draws shift if the RNG order
   changes upstream.
4. **Deterministic NLopt with no spread desired** — `nl_sbplx` and `nl_bobyqa`
   are deterministic, so `single` + `chains > 1` gives N identical optimisations
   and the chain-agreement gate is uninformative; only use this when you
   explicitly want a single optimisation from a known seeded point.

**Per-stage independence.** Scout and refine can use different `init` modes (LHS
for basin-finding in scout, `single` in refine to converge from scout's MLE).
The CLI `--init` flag requires `--stage` for the same reason — it's
stage-scoped.

**`camdl profile`** dispatches the same way at each grid cell:
`--starts N --init lhs` draws N stratified per-cell starts across the non-focal
estimated parameters; the focal parameters stay pinned to the grid point.
`--init single` reproduces the historical "every start at the same point, IF2
RNG provides the spread" behaviour.

### Out-of-sample validation

Add a `[holdout]` section to fit.toml with holdout data files:

```toml
[data]
weekly_cases = "data/cases_train.tsv"

[holdout]
weekly_cases = "data/cases_holdout.tsv"
```

Scout and refine only see `[data]` — holdout is structurally unreachable during
parameter estimation. Validate runs the particle filter on train + holdout and
reports separate logliks:

```
train loglik:   -4200.3 (780 obs)
holdout loglik: -1615.1 (316 obs)
```

Use `camdl data split` to produce train/holdout files:

```bash
camdl data split data/cases.tsv --at-time 5474
```

### Prediction quantiles

The pfilter trace includes both observation-space and state-space prediction
quantiles:

- `obs_mean`, `obs_q05`, `obs_q50`, `obs_q95` — full predictive distribution
  (process + observation noise). Data should fall inside the 5-95 ribbon ~90% of
  the time.
- `state_mean`, `state_q05`, `state_q50`, `state_q95` — latent state quantiles
  mapped through the observation model mean. Process uncertainty only.

Both are on the observation scale (reported cases, not latent recoveries). The
gap between the obs and state ribbons shows the observation model's contribution
to uncertainty.

### Pfilter replicates

```bash
camdl pfilter model.camdl --params mle.toml --data d.tsv \
    --replicates 100 --output logliks.tsv
```

Runs N independent particle filters at different seeds. Reports
`loglik = -3804.9 ± 5.2 (100 replicates, N=5000)`.

See `docs/camdl-inference-spec.md` for the full specification.

### Saving final particle states

For prediction workflows, `camdl pfilter --save-final-state` writes the particle
ensemble at the last observation time:

```bash
camdl pfilter model.camdl --data train.tsv --params mle.toml \
    --particles 5000 --save-final-state final_particles.tsv
```

Output is a TSV with one row per particle, columns for each compartment and flow
accumulator. This enables forward simulation from the filtered state without
re-running the particle filter.
