# PMMH proposal adaptation: a floor, a noise-aware target, and one shared Robbins-Monro

Status: proposed Related: gh#347 (the deterministic-MH deadlock this shares
machinery with)

## The problem

A PMMH chain can shrink its proposal to zero and stop moving, silently. The run
completes, writes draws, and reports a bad R-hat that looks like slow mixing
rather than a chain that stopped sampling.

Observed on a national Ebola fit: six chains, 19,200 particles, 5,000
iterations. Two chains reached sustained 0.0% acceptance with no accepted move
at all; one saw its per-parameter step fall by a factor of 1,157 over the run
and was still falling at the last block.

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

    a(sigma) = 2 * Phi(-sigma / sqrt(2))

with `sigma` the standard deviation of the estimated log-likelihood and `Phi`
the standard normal CDF. Verified by direct Monte Carlo of the noise-only chain
(sigma=1: 0.4805 measured against 0.4795 predicted; sigma=2: 0.1561 against
0.1573).

If `a(sigma)` sits below the target, **no step size reaches the target**, the
Robbins-Monro recursion has no root, and `log lambda` decreases without bound.
Measured: lambda = 5.9e-17 after 200k steps at 15% acceptance.

The target is `0.234 + 0.206/d`. The crossover is therefore d-dependent — at d =
17 the target is 0.2461 and the crossover is sigma = 1.640.

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

## Fix 1: bound the scale

Project `log_scale` onto `[-10, 5]` after every update, and emit an end-of-run
warning counting steps spent on the floor.

The bound is not the same number as `pgas`'s `[-20, 5]` because the quantities
differ: `pgas`'s `log_proposal_sd` is an absolute per-parameter SD, while
`pmmh`'s `log_scale` is a relative multiplier on a proposal already scaled by
`2.38^2/d`, so `lambda = 1` is the optimum by construction. At -20 the floor is
too low to bind before the chain is dead; at -10 gh#347's deadlock test still
recovers to lambda = 4.5e-5, well inside its bar.

The warning is load-bearing rather than decorative. A floor bounds the damage
without repairing the chain: a run that sat on it did not explore its posterior,
and the output must say so rather than presenting a stalled chain as a fit.

This does not rescue a chain in a high-noise region. At sigma = 6 the limiting
acceptance is 4e-5 at every step size; flooring the scale stops it reaching zero
and does not make the chain move.

## Fix 2: a noise-aware target acceptance

The target `0.234 + 0.206/d` is the optimal-scaling result for random-walk
Metropolis with an exact likelihood. It is the wrong target for a
pseudo-marginal chain, and demanding it is what removes the recursion's root.

Measured: forcing the target to 0.07 at sigma=2 removes the collapse entirely
(lambda ends at 1.18, Haario diagonal 0.99, against 1.4e-4 and 0.11 today). But
at sigma=0 and sigma=1 the same target overshoots to lambda 1.9 and 1.6, so 0.07
cannot simply become the default.

The target must therefore be a function of the estimated likelihood noise:
deterministic `mh` keeps `0.234 + 0.206/d`; pseudo-marginal PMMH derives its
target from a measured `sigma`. Sherlock, Thiery, Roberts and Rosenthal (2015),
_Annals of Statistics_ 43(1):238-275, give the optimal pseudo-marginal
configuration as sigma = 1.8 with an acceptance rate substantially below the
exact-likelihood 23.4%.

camdl already measures `sigma` in the PF-variance preflight
(`cli/src/fit/pmmh.rs`), so the input exists. Two gaps: it is measured only at
the base parameter vector, and `sigma` varies strongly over parameter space —
measured at 1.13 at a posterior median and 6.04 at a chain five posterior
standard deviations out, a factor of 5.3. A target set once from a base-point
`sigma` will be wrong for a chain that wanders.

## Fix 3: the preflight blesses runs that collapse

`cli/src/fit/pmmh.rs` prints a green `PF variance OK (target: 1-3)` for any
`sigma` in [0.5, 5.0]. Collapse begins near the d-dependent crossover, which is
1.64 at d=17. The band should be derived from the crossover rather than fixed,
and the check should say which side of it the run sits on.

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

Fixes first, refactor second. The fixes are small, urgent and reviewable in
isolation; the refactor touches two samplers across roughly 2,100 lines and
would bury them. More importantly, landing the fixes first gives the refactor a
test suite to refactor _against_ — the reverse order restructures code whose
failure mode has no coverage, which is how this defect survived gh#347.

## Decisions for the maintainer

1. **What the four blocked tests should assert.** The reproduction branch
   carries tests demanding the scale hold at sigma 2 to 5. No bound can satisfy
   them, because the target acceptance is itself unreachable there.
   Recommendation: rewrite them to assert the floor holds and the warning fires,
   and drop the Haario-ratchet test, which asserts both `1/sqrt(n)` decay and
   less than a 2x fall over a 32-fold range of n — no implementation satisfies
   both.
2. **Whether fix 2 ships with fix 1 or separately.** Recommendation: separately.
   Fix 1 is a bound with no behavioural subtlety; fix 2 changes the target
   acceptance and therefore every pseudo-marginal fit's trajectory.
3. **Whether `sigma` is re-measured during the run.** A target set once from a
   base-point `sigma` is wrong for a chain that wanders into the tail, which is
   the case that motivated this. Recommendation: measure per chain at the
   adaptation-window boundary, and report it.
4. **Scope of the refactor.** Recommendation: items 1 to 4 above, deferring the
   CLI driver split (item 5) until the sampler side has settled.

## What this does not address

A chain in a genuinely high-noise region is not saved by any of this. The
remedies there are correlated pseudo-marginal (already implemented, the `rho`
stage key), more particles, or a prior that puts less mass on regions where the
filter degenerates. Those are modelling and configuration choices, not sampler
defects.
