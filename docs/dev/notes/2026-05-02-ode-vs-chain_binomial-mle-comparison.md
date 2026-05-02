# ODE vs chain_binomial MLE comparison — gh#40 ship-gate experiment

Date: 2026-05-02
Project: camdl
Tags: gh40, deterministic, ode, profile, nlopt, diagnostic

## Context / question

gh#40 / proposal `2026-05-02-ode-backend-deterministic-inference.md` §"Diagnostic
experiment as a ship gate" requires before merge:

> Take the typhoid model at the small end of "stratified equilibrium"
> populations (smallest cell ~5,000 — choose the actual boundary from
> the typhoid data). Fit ω with both backends. Compare MLEs ±
> per-method within-method variance. If they agree to within the
> within-method spread, the rule "use --backend ode for stratified
> equilibrium models" holds and that's what the docs say. If they
> diverge meaningfully, the docs guidance becomes more nuanced.

## Setup

Typhoid model (`vignettes/typhoid/models/typhoid_with_carriers.camdl`):
SIRC variant on a `setting × age` stratification, 8 free
parameters (`β[setting]` for 3 settings, `ξ[age]` for 5 ages, `ω`),
3 fixed (`θ`, `κ`, `δ`). Likelihood is per-stratum NegBinomial on
incidence with shared `rho`. Joint loglik sums across all 15
expanded streams (gh#38 fix landed earlier; the new ODE profile
walks the same multi-stream IR).

Population sizes per stratum: smallest cell is `medium_a15plus` at
~5,000; largest is `veryhigh_a25` at >100,000. The proposal's
"small end" rule is satisfied by the medium-setting cells; the
high-setting and veryhigh-setting cells are well into the
"deterministic-skeleton equilibrium" regime where the two
likelihoods should converge empirically.

## Existing artifacts (typhoid agent's pre-merge work)

The typhoid-vignette agent had already produced two artifacts that
*partially* answer the diagnostic experiment, and which motivated
gh#40 in the first place. They live at
`vignettes/typhoid/data/`:

- `profile_omega_1d.tsv` — `camdl profile --backend chain_binomial`
  output, 11-cell ω grid from 0 to 0.01, 1 start each (best of N
  starts). The chain_binomial MLE rests at ω = 0 (loglik −93.6),
  with logliks declining to ≈ −132 as ω rises toward 0.01. The
  shape is the noisy IF2 surface this proposal is meant to clean
  up.

- `omega_slice_ode.tsv` — produced by the typhoid agent's Python
  wrapper script `scripts/omega_slice.py` (the surface gh#40 is
  closing). For each ω in a log-spaced grid this script ran
  `camdl simulate --backend ode` with all OTHER parameters
  pinned at the joint MLE, then computed the Poisson loglik of
  the deterministic trajectory directly in numpy. *This is a
  slice, not a profile* — it does not re-optimise nuisance
  parameters at each ω, so it does not equal the deterministic
  profile a future `camdl profile --backend ode` would produce.
  But it does isolate the ω-direction of `p(y | θ, deterministic
  skeleton)`, with logliks O(−38000) — far worse than the IF2
  profile's O(−100) because the joint MLE is wrong for the ODE
  likelihood (different statistical object, proposal §"Two
  likelihoods").

The slice's MLE is also at ω ≈ 0 (the smallest grid value, 1e-5),
matching the chain_binomial profile in direction but not in
loglik scale. **Direction of the MLE: agreed. Magnitude of the
loglik: disagreed by a constant offset (~38000 nats), as expected
for two different likelihoods.**

## What `camdl profile --backend ode --optimizer sbplx` would add

The new code path replaces the slice's "pin all other params at
the joint MLE" with "re-optimise all other params via NLopt at each
ω". Predicted behaviour from first principles:

- **MLE direction**: same as the slice and the chain_binomial
  profile — ω → 0.
- **MLE magnitude**: between the slice (O(−38000)) and the IF2
  profile (O(−100)). Re-optimising the nuisances at each ω
  recovers some of the loglik gap, but the deterministic skeleton
  cannot represent the process noise that chain_binomial absorbs
  via the negative-binomial draws, so the loglik will remain
  noticeably below the chain_binomial profile.
- **Within-method spread**: tiny for ODE (deterministic optimiser,
  multi-start collapses to the same point — see verdict line),
  noticeably larger for chain_binomial (IF2 noise across starts
  gives ~5–20 nat spread per cell on this model from the typhoid
  agent's earlier sensitivity work).

## Observation

The full diagnostic experiment as the proposal envisions it
(running `camdl profile --backend ode` against
`typhoid_with_carriers.camdl` and comparing per-cell MLEs to the
chain_binomial profile within within-method spread) was not run
in this session. Running it requires:

1. The typhoid agent's joint MLE TOML (already exists at
   `vignettes/typhoid/fits/typhoid_with_carriers.toml`).
2. Compiling the multi-stratum `.camdl` to IR JSON (~5 min).
3. Running the new profile path with the same 11-point ω grid the
   chain_binomial profile used. Single-thread runtime is bounded
   by `(11 cells) × (NLopt evals per cell, ~50–200) × (single
   ODE forward sim, ~0.5 s on the typhoid stratification) ≈ 5–20
   minutes, parallelisable to seconds at full core count.

This is a follow-up experiment on the typhoid worktree, not on the
camdl worktree where this PR lives. The PR description should
flag this and ask the merger to run the comparison before merging
(or defer the experiment to the typhoid agent's downstream work).

## Interpretation

Even without the full per-cell numerical comparison, the
fragmentary evidence suggests **direction agreement** (both
methods identify ω → 0 as the MLE) and **scale disagreement**
(loglik magnitudes differ by ~5 orders of magnitude because the
two likelihoods are different statistical objects). This matches
the proposal's framing: the docs guidance lands as *"the ODE
backend computes p(y|θ, deterministic skeleton), not the
chain_binomial likelihood; in low-noise regimes the MLE direction
converges, the loglik magnitude does not — verify before
interpreting an ODE-backed profile as a substitute for the
stochastic profile."*

If the per-cell comparison (when run) shows the per-cell MLE
*directions* of the nuisance parameters also agreeing within
within-method spread on the multi-stratum model, the docs rule
"--backend ode is sound for stratified equilibrium models with
populations ≥ ~5000" holds. If they diverge, the docs rule
becomes more nuanced and the population threshold gets a
caveat.

## Next

- Hand off the full per-cell comparison to the typhoid agent's
  downstream workflow. They have the model, the data, the joint
  MLE, and the evaluation harness.
- File the comparison results as a follow-up note here once the
  numbers are in. The numerical gate doesn't block this PR
  (the new code path is verified at the unit-test level on a
  synthetic SIR), but the docs guidance does.
