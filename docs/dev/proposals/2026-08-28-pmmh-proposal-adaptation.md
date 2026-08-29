# PMMH proposal adaptation: a floor, a noise-aware target, and one shared Robbins-Monro

Status: fix 1 (a9a6355d), fix 4 (63dd20c2) and fix 3 implemented; fix 2 deferred
to gh#767.

Related: gh#347 (the deterministic-MH deadlock this shares machinery with),
gh#764 (persisting the measured spread — landed with fix 3).

## The problem

A PMMH chain can shrink its proposal to zero and stop moving, silently. The run
completes, writes draws, and reports a bad R-hat that looks like slow mixing
rather than a chain that stopped sampling.

Observed on a national Ebola fit: six chains, 19,200 particles, 5,000
iterations. Two chains reached sustained 0.0% acceptance with no accepted move
at all; one saw its per-parameter step fall by a factor of 1,157 over the run
and was still falling at the last block.

## Terms

- **`d`** — the number of estimated parameters, i.e. the dimension of the vector
  being sampled.
- **`sigma`** — the standard deviation of a single estimated log-likelihood
  `log L-hat` at a fixed parameter vector. This is what the PF-variance
  preflight measures and what "aim for a log-likelihood standard deviation near
  1" refers to.
- **`s`** — the standard deviation of the _difference_ between two evaluations,
  `log L-hat(theta') - log L-hat(theta)`. This is the quantity that enters the
  Metropolis ratio, so it is the one that governs acceptance. For independent
  evaluations `s = sigma * sqrt(2)`; under correlated pseudo-marginal (`rho`
  set) the two evaluations share most of their randomness and `s` is smaller.
- **`lambda`** — the Robbins-Monro global proposal scale, `exp(log_scale)`.
- **`a*`** — the target acceptance rate the adaptation drives toward.

## Mechanism

The proposal is `theta' = theta + lambda * L * z`, where
`L L^T = (2.38^2/d) *
Cov + eps*I` is the Haario shape term and
`lambda = exp(log_scale)` is a Robbins-Monro scalar driven toward a target
acceptance rate.

`AdaptiveProposal::adapt_scale` applies
`log_scale += gamma * (accepted -
target)` with `gamma = (step+1)^-0.6` and **no
bound**. That is sound when a step size exists that achieves the target. Under a
pseudo-marginal likelihood it often does not.

With an estimated likelihood the chain sticks on lucky over-estimates: it lands
where the particle filter happened to return a high `log L-hat`, and every
subsequent proposal is compared against that inflated value and rejected. In the
small-step limit the acceptance rate tends to

    a_ceiling = 2 * Phi(-sigma / sqrt(2))  =  2 * Phi(-s / 2)

with `Phi` the standard normal CDF. The two forms are the same number written in
the two spreads defined above; the `s` form is the one to implement against,
because it is scheme-agnostic (see fix 3). Verified by direct Monte Carlo of the
noise-only chain, twice independently:

| sigma | formula | run 1  | run 2  |
| ----- | ------- | ------ | ------ |
| 1.0   | 0.4795  | 0.4805 | 0.4807 |
| 2.0   | 0.1573  | 0.1561 | 0.1555 |

If `a(sigma)` sits below the target, **no step size reaches the target**, the
Robbins-Monro recursion has no root, and `log lambda` decreases without bound.
Measured: lambda = 5.9e-17 after 200k steps at 15% acceptance.

The target is `0.234 + 0.206/d`, the optimal-scaling result for random-walk
Metropolis with an _exact_ likelihood. The crossover — the `sigma` at which the
ceiling falls below it — is nearly flat in `d`:

| d  | target `a*` | crossover sigma |
| -- | ----------- | --------------- |
| 1  | 44.0%       | 1.09            |
| 6  | 26.8%       | 1.57            |
| 17 | 24.6%       | 1.64            |
| 50 | 23.8%       | 1.67            |

It asymptotes to 1.683, so for any realistic model collapse begins around
`sigma ~ 1.6`.

**The defect is the target, not the noise level.** Sherlock, Thiery, Roberts &
Rosenthal (2015, _Ann. Statist._ 43(1):238-275) prove the pseudo-marginal
random-walk Metropolis is optimally efficient at noise variance 3.283 —
`sigma = 1.812` — with an optimal acceptance rate of 7.001%. At that `sigma` the
ceiling is 20.0%, so the literature's own operating point is comfortably
_attainable_. What is unattainable there is camdl's 24.6% target at d = 17.
Running at the recommended noise level is fine; demanding the exact-likelihood
acceptance rate while doing so is not.

Because `lambda` multiplies the whole factor including `eps*I`, an unbounded
`lambda` also voids the only floor Haario's algorithm has. The covariance
recursion itself is not the defect: `eps = 1e-6` floors the shape term's SD at
1e-3 by construction, and over a sigma=2 run it fell only 0.44 to 0.11.

`pgas.rs` has guarded against this since gh#347 and `pmmh.rs` has not, despite a
comment in the CLI driver recording that the two share this machinery.

## Why this was not caught

gh#347's fix landed _in_ `pmmh.rs` and was verified on the deterministic ODE
`mh` arm plus unit tests driving `adapt_scale` with forced accept/reject
sequences. Both are noise-free. The coupled loop — adaptation reacting to an
acceptance rate that is itself a function of likelihood noise — was never
exercised. The regime where the recursion loses its root is exactly the regime
no test covered.

## Fix 1: bound the scale — implemented (a9a6355d)

Project `log_scale` onto `[-10, 5]` after every update, count the steps spent on
the floor, and emit an end-of-run warning naming the knob that helps.

The bound is not the same number as `pgas`'s `[-20, 5]` because the quantities
differ: `pgas`'s `log_proposal_sd` is an absolute per-parameter SD, while
`pmmh`'s `log_scale` is a relative multiplier on a proposal already scaled by
`2.38^2/d`, so `lambda = 1` is the optimum by construction. At -20 the floor is
too low to bind before the chain is dead; at -10 gh#347's deadlock test still
recovers to `lambda = 4.5e-5`, well inside its bar.

The warning is load-bearing rather than decorative. A floor bounds the damage
without repairing the chain: a run that sat on it did not explore its posterior,
and the output must say so rather than presenting a stalled chain as a fit.
`PMMHResult.steps_at_scale_floor` carries the count so a caller can gate on the
condition instead of scraping stderr.

**What it does not cover.** The warning fires only if the floor is _reached_.
`log lambda` drifts as `-(a* - a) * T^0.4 / 0.4` over `T` steps, so at the
literature-optimal `sigma = 1.812` (a gap of 4.6 percentage points):

- `T = 5,000`: the closed form gives `lambda = 0.031`, but that holds `a` at the
  `lambda -> 0` ceiling; at `lambda = 1` acceptance is 8.0%, not 20.0%, so the
  early gap is 16.6 points and the real drift is faster. A coupled d = 17 run of
  the actual recursion measures `lambda = 0.020` (median of 9 seeds). Treat
  0.031 as an upper bound, and note `sd(log lambda) ~ 0.87` from the
  stochastic-approximation noise alone — a factor of 2.4 either way across
  seeds, so no point value here is meaningful without its spread.
- The floor is first reached at a median of **~61,000 steps** (12 runs, range
  40,067-91,003).

So a proposal 30-50x narrower than the covariance-optimal scale, with expected
squared jump distance down about **400x** — not the `lambda^-2` factor of 1000x,
because acceptance _rises_ from 8.0% to 20.0% as `lambda` falls and partly
offsets it. The floor is never touched below ~40,000 steps, so a normal-length
run at the recommended noise level under-mixes with nothing warning. Fixes 3 and
4 close that gap without needing a new target.

## Fix 2: a noise-aware target acceptance — deferred to gh#767

The target `0.234 + 0.206/d` is the exact-likelihood optimum and is the wrong
target for a pseudo-marginal chain; demanding it is what removes the recursion's
root. Measured on the synthetic harness, forcing the target to 0.07 at
`sigma = 2` removes the collapse entirely (`lambda` ends at 1.18, Haario
diagonal 0.99, against 1.4e-4 and 0.11), but the same target overshoots at
`sigma = 0` and `sigma = 1` to `lambda` 1.9 and 1.6, so 0.07 cannot become the
default.

**The rule is available in closed form.** Sherlock et al.'s limiting model has
the log-ratio distributed `N(-v/2, v)` with `v = l^2 + 2*sigma^2` for scaling
`l`; maximising expected squared jump distance over `l` at _fixed_ `sigma` is
the same optimisation with one argument frozen, and the paper's own finding that
"the optimal scaling is insensitive to the noise" is the statement that this
conditional optimum is well-defined. Reconstructing the model reproduces the
published joint optimum (variance 3.2833 against the paper's 3.283, acceptance
6.9996% against 7.001%) and the classical exact-likelihood limit (`l* = 2.3812`,
`a* = 23.38%`), which is the check that it is the right model:

| sigma | optimal `l` | optimal target `a*(sigma)` | ceiling `2*Phi(-s/2)` |
| ----- | ----------- | -------------------------- | --------------------- |
| 0.0   | 2.381       | 23.38%                     | 100%                  |
| 1.0   | 2.464       | 15.54%                     | 47.95%                |
| 1.812 | 2.562       | 7.00%                      | 20.01%                |
| 2.0   | 2.582       | 5.55%                      | 15.73%                |
| 3.0   | 2.662       | 1.23%                      | 3.39%                 |

`a*(sigma)` sits below the ceiling at every `sigma` by construction, so the
recursion always has a root. This is a one-line optimisation, not a research
programme.

**Deferred anyway, and for one reason only: transfer.** The table above is a
limiting result for a target with i.i.d. components and Gaussian estimator
noise. A real posterior's `sigma` varies fivefold across the space — 1.13 at a
posterior median, 6.04 five posterior standard deviations out — so a target set
from a base-point `sigma` is wrong for exactly the chain that wanders, which is
the case that motivated this. That is a design question about _which_ `sigma`
the target should track, not about what the target should be given one.

A target rule changes the trajectory of every pseudo-marginal fit. gh#767
carries the evidence and the acceptance criteria; the four `#[ignore]`d tests in
`pmmh_scale_collapse.rs` are its specification.

## Fix 3: a preflight that computes the ceiling instead of guessing a band

Implemented in `cli/src/fit/pf_noise.rs`; the two points the spec below left
open are settled at the end of this section.

`cli/src/fit/pmmh.rs` printed a green `PF variance OK (target: 1-3)` for any
`sigma` in `[0.5, 5.0]`. Collapse begins around 1.6, so the check says "OK"
across a range that is mostly past the point where the adaptation loses its
root. That is worse than silence.

**Measure `s`, not `sigma`.** The preflight currently takes 20 replicates of a
single `log L-hat` and reports their standard deviation. Two changes:

1. Report the acceptance ceiling `2 * Phi(-s / 2)` against this run's own target
   `0.234 + 0.206/d`, and say which side of it the run sits on. No tuned band,
   no magic constants — the computation is the message.
2. Measure `s` under the scheme the run actually uses. This matters most for
   correlated pseudo-marginal: the preflight calls `run_quick_pfilter`, the
   _plain_ bootstrap filter, for all 20 replicates regardless of whether `rho`
   is set. But CPM exists precisely to shrink the log-ratio spread by reusing
   randomness between evaluations, so the plain-filter number overstates the
   noise that governs a CPM run's acceptance — at `sigma = 2` with an induced
   correlation of 0.9 between successive estimates, the true ceiling is 65%
   rather than the 15.7% the plain measurement implies. Evaluate with the
   pre-drawn randoms, then with their Crank-Nicolson update, and take the spread
   of the _differences_; for plain PMMH the same procedure gives
   `s = sigma * sqrt(2)` and nothing changes.

**Three constraints on that measurement, each of which came out of review.**

- **Never derive `s` from `rho`.** The identity `Var = 2*sigma^2*(1-rho_ll)`
  holds for `rho_ll`, the correlation of the log-likelihood _estimates_, which
  is not the Crank-Nicolson parameter unless the estimator is a linear Gaussian
  functional of the auxiliary variables. Measured on a realistic skewed unbiased
  estimator, a CN parameter of 0.90 induced `rho_ll = 0.81`, and substituting
  the CN parameter understated `s` by 36%. Measure `sd(difference)` directly.
- **The ceiling assumes `log L-hat` is approximately Gaussian, and the error is
  one-sided.** Under skew the predicted ceiling is _optimistic_ — the wrong
  direction for a check whose job is to catch collapse. The assumption is safe
  once the noise is a sum over many observation times (measured relative error:
  54% at 1 observation, 9% at 5, under 1% at 20), so state it rather than
  presenting the ceiling as unconditional.
- **Measured at the base point, this is a best case.** The correlation the run
  realises also depends on `theta` moving, and the achievable CPM correlation
  degrades with the dimension of the auxiliary variable (Deligiannidis, Doucet &
  Pitt 2018, _JRSS-B_ 80(5):839-870). Take the second evaluation at a `theta'`
  drawn from the initial proposal rather than at fixed `theta`, so the number
  reflects the scheme _and_ the step.

**Cost.** Spreads of differences need pairs: 20 pairs is 40 filter evaluations,
double the current preflight. At 19,200 particles that is not free, and it is
the same order of cost this proposal declines to pay for per-window `sigma`
re-measurement (decision 3). Twenty pairs is the recommendation; fewer trades
directly against the interval below.

**Warn and proceed** when the ceiling is below the target. A user may be running
a deliberately cheap exploratory fit, and refusing an expensive run at preflight
is a stronger action than the diagnosis warrants. The message states the
ceiling, the target, and that `lambda` will fall for the whole run.

**Report the spread's own uncertainty.** Twenty pairs gives the standard
deviation a standard error of roughly `s / sqrt(2 * 19)`, about 16% — one
standard error, so roughly 68% coverage. Near a crossover of 1.64 that is +/-
0.26 at one SE and +/- 0.52 for a 95% read. The check reports the interval,
labels which it is, and says the verdict is unresolved when the interval
straddles the crossover rather than picking a side.

`sigma`, `s`, the particle count and the replicate count are persisted with the
stage artifact (gh#764); a spread without its particle count is meaningless,
since `sigma` scales as `1/sqrt(N)`.

**One `theta'`, reused across the pairs.** The constraint above says the second
evaluation is taken at a `theta'` drawn from the initial proposal; it does not
say whether that draw is repeated. It is not. The pair difference is
`log L-hat(theta') - log L-hat(theta)`, so with `theta'` fixed the true
log-likelihood difference is a constant offset and the spread of the differences
is estimator noise alone — the quantity the ceiling is a function of. Redrawing
`theta'` per pair would fold the curvature of the log-likelihood surface into
`s`, and the ceiling would then be a statement about the posterior rather than
about the filter. The cost is that the measurement is conditional on the one
`theta'` drawn, and the reported standard error does not cover that
conditioning.

**A consequence worth stating: `s` is not `sigma*sqrt(2)`, even for plain
PMMH.** The two evaluations sit at different `theta`, and the filter's noise
level varies across the parameter space, so `s^2` is the _sum_ of the two
points' variances. Measured on a two-parameter SIR with 20 daily observations at
200 particles: `sigma` 4.07 at the base point, `s` 9.40 across the step,
implying a `sigma` of 8.5 where the proposal lands. A ceiling computed from that
errs toward saying the chain cannot accept, which is the safe direction for this
check.

## Fix 4: adaptation stops at the end of warm-up

`adapt_scale` runs from step 0 to the last step of the chain. `adapt_start`
gates only the covariance shape term, not `lambda`. So the draws that are kept
are produced while the proposal is still shrinking, and the drift is bounded by
the _run length_ rather than by the warm-up budget.

Freeze `lambda` and the Haario shape term at the end of warm-up. Two things
follow:

- The drift is capped by the warm-up length rather than the run length. At
  `sigma = 1.812` with a 500-step window, `lambda` bottoms at 0.25 instead of
  0.02 — still narrow, but bounded, stable, and diagnosable at a single point.
- The sampling phase becomes a fixed transition kernel, so the kept draws are
  exactly invariant for the posterior rather than relying on the
  diminishing-adaptation and containment conditions (Roberts & Rosenthal 2007)
  that a continuously-adapting chain needs. This is what Stan and most
  production samplers do.

**This is a no-op under the shipped defaults, and that has to be fixed with
it.** `DEFAULT_BURN_IN` is 5,000 (`cli/src/fit/pmmh.rs`) and the motivating
Ebola run was 5,000 iterations, so `burn_in == n_steps` and freezing "at the end
of warm-up" freezes at the last step — no change at all for exactly the run this
proposal is about. Two consequences: the warm-up boundary should be an explicit
field rather than an overload of `burn_in`, whose default is already the whole
run; and whatever `DEFAULT_BURN_IN` becomes is a separate decision that moves
every PMMH run's output and must not ride along silently.

**The tradeoff, stated.** The current design is theoretically valid —
diminishing gain plus fix 1's bound is exactly containment — and gh#347's
deadlock rescue depends on `lambda` adapting early, which a warm-up window
preserves. But this changes the trajectory of every PMMH run, including healthy
ones, so it is a behaviour change rather than a bug fix and lands with that
stated.

Alongside it: report the end-of-run `lambda` and flag when it has moved far from
unity. A `lambda` of 0.02 means the proposal ended 50x narrower than the
estimated covariance, which is the silent case fix 1's floor warning misses.
This is a diagnostic, changes no trajectory, and closes the blind spot on its
own.

## The refactor, and where the seam actually is

The natural reading is "extract the shared MH machinery from `pmmh.rs`". That is
too broad. The two samplers adapt genuinely different objects:

- `pmmh.rs` proposes a **joint block move**: a dense Haario covariance with one
  global scalar (`m2`, Cholesky factor, `log_scale`).
- `pgas.rs` proposes **coordinate-wise MH-within-Gibbs**: independent
  per-parameter SDs (`log_proposal_sd: Vec<f64>`), per tempering rung, with
  per-parameter acceptance indicators and no covariance at all.

Unifying the whole proposal would mean an abstraction that erases that
difference. The genuinely shared substrate is narrower and is exactly the part
that diverged: **the Robbins-Monro update and its guards.**

    log_x += gamma(step) * (accept_indicator - target)
    log_x  = log_x.clamp(lo, hi)
    (and: adaptation stops after the adaptation window)

That is identical logic in both files, written twice, with different guards. One
shared implementation, with the clamp and the adaptation-window check inside it,
makes this bug class unrepresentable rather than merely fixed. Both call sites
keep their own shape estimation.

What the refactor should include:

1. Extract the Robbins-Monro update into one place, carrying the bound and the
   adaptation-window check. Both call sites adopt it. Bounds stay per-call-site
   parameters, because the quantities have different units.
2. Move `AdaptiveProposal` out of `pmmh.rs` into its own module. It is generic
   MH machinery used by two samplers and named for one of them.
3. Move `mcmc_ess` out. It is a generic MCMC diagnostic in a sampler-specific
   file.
4. Rename `run_pmmh`, which is named for one of its two callers. Its likelihood
   argument is already a closure, so the algorithmic core is genuinely generic —
   this is a naming and placement fix, not a restructuring.
5. Split the CLI driver, or reduce `is_ode_mh` to a single explicit seam rather
   than a branch threaded through 1,216 lines.

What the refactor must preserve:

- Byte-identical results for any configuration whose behaviour is not
  intentionally changed. A refactor that silently moves numbers is worse than no
  refactor.
- The existing suites: `--test pmmh` (13), `adaptive_scale_tests` (3),
  `pmmh_hierarchical` (4), `correlated_pf_finite` (9),
  `gh226_inf_loglik_backstop` (3), and gh#347's deadlock test.
- `pgas.rs` is a high-risk surface. Its adaptation is per-rung and
  per-parameter; the shared update must not quietly change its semantics.

## Sequencing

1. **gh#764** — persist the measured spread with its particle and replicate
   counts. Everything below reads it, and a spread without its particle count is
   not a number anyone can act on.
2. **Fix 3** — the preflight. Small, no design tension, and it stops the check
   from blessing runs that provably cannot sample.
3. **Fix 4** — freeze adaptation at the end of warm-up, and report the
   end-of-run `lambda`. A behaviour change, so it lands on its own.
4. **gh#767** — the noise-aware target, once the efficiency sweep exists.
5. **The refactor**, last.

Fixes before refactor. The fixes are small, urgent and reviewable in isolation;
the refactor touches two samplers across roughly 2,100 lines and would bury
them. More importantly, landing the fixes first gives the refactor a test suite
to refactor _against_ — the reverse order restructures code whose failure mode
has no coverage, which is how this defect survived gh#347.

## Decisions, settled

1. **What the four blocked tests should assert.** They are kept as the
   acceptance criteria for fix 2 rather than rewritten: renamed to state the
   property they demand, marked `#[ignore]` with the reason, and joined by new
   tests for what the bound does deliver. The Haario-ratchet unit test is
   dropped — it asserts both `1/sqrt(n)` decay and less than a 2x fall over a
   32-fold range of n, and no implementation satisfies both.
2. **Whether fix 2 ships with fix 1.** Separately, and in the event fix 2 is
   deferred entirely to gh#767: it needs an efficiency sweep that does not exist
   yet, and a target rule fitted on a Gaussian toy would trade a diagnosable
   failure for an undiagnosable one.
3. **Whether `sigma` is re-measured during the run.** Not in fix 3. The
   preflight measures at the base point, reports that scope explicitly, and
   reports its own uncertainty. Re-measuring per chain at each adaptation-window
   boundary costs 20 extra filter evaluations per chain per window — real money
   at 19,200 particles — and it is only worth paying once a target actually
   consumes the number, which is gh#767's problem.
4. **What happens when the ceiling is below the target.** Warn and proceed. The
   user may be running a deliberately cheap exploratory fit, and refusing an
   expensive run at preflight is a stronger action than the diagnosis warrants.
5. **Scope of the refactor.** Items 1 to 4 below, deferring the CLI driver split
   (item 5) until the sampler side has settled.

## What this does not address

A chain in a genuinely high-noise region is not saved by any of this. The
remedies there are correlated pseudo-marginal (already implemented, the `rho`
stage key), more particles, or a prior that puts less mass on regions where the
filter degenerates. Those are modelling and configuration choices, not sampler
defects.
