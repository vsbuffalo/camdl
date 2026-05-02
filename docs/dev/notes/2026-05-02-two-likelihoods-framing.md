# Two likelihoods, not one optimiser for one likelihood — gh#40 docs framing

Date: 2026-05-02
Project: camdl
Tags: gh40, ode, profile, docs, user-facing

## Context

`camdl profile --backend ode --optimizer sbplx` ships in gh#40. The
load-bearing user-facing point — the one the camdl-book chapter
will need to make in prose — is that **the choice of `--backend`
selects between two different statistical objects, not two
implementations of the same one**.

This note is the source of truth for that framing. CLI `--help`
text and the eventual book chapter on `camdl profile` should both
reflect it.

## The framing in one paragraph

> When a model is fit with `--backend chain_binomial`, the model's
> likelihood is *defined by* its stochastic forward kernel: the
> particle filter gives an unbiased estimator of `p(y | θ)` where
> the process noise is part of the generative model. When the same
> `.camdl` file is fit with `--backend ode`, we are computing a
> *different* likelihood — `p(y | θ, deterministic skeleton)` —
> for the same model. These are not the same statistical object.
> They will give different MLEs, different identifiability
> properties, and different uncertainty (Wald CIs from a Hessian
> under-cover relative to PMMH for stochastic models). In low-noise
> regimes (large per-cell populations, near-deterministic
> trajectories) the two likelihoods converge empirically. In
> high-noise regimes they don't. The right user-facing rule is
> therefore not *"`--backend ode` is faster when populations are
> large"* but rather *"`--backend ode` answers a different
> scientific question; in low-noise regimes the answers converge —
> verify, don't assume."*

## Where this surfaces in the CLI

`camdl profile --help` carries a condensed version on the
`--backend` flag itself:

> `chain_binomial` (default) runs IF2 on a stochastic
> chain-binomial forward model — the same likelihood `camdl fit`
> uses today. `ode` runs deterministic ODE forward sims + NLopt
> optimisation, computing `p(y|θ, deterministic skeleton)` —
> a *different* likelihood (gh#40 proposal §"Two likelihoods").
> In low-noise regimes (large per-cell populations,
> near-deterministic trajectories) the two converge empirically;
> verify, don't assume.

## Where this surfaces in the book

Not yet — gh#40 ships the code path; the book chapter is a
separate downstream task on the camdl-book worktree. When the
typhoid chapter is updated to use `--backend ode` for the
deterministic profile, the text should:

1. State the two-likelihoods framing explicitly at the point
   where `--backend ode` first appears.
2. Cite the gh#40 proposal as the design reference.
3. Show — empirically — that on the typhoid model the two
   profiles agree on the MLE direction, with a numerical
   comparison if the diagnostic experiment from the proposal
   has been run.

## Algorithm choice rule (also user-facing)

`--optimizer` defaults to `sbplx` (NLopt's `LN_SBPLX`, a robust
Nelder-Mead variant). The rule the help text bakes in:

- **Sbplx** is the default *because* compartmental likelihoods
  are smooth in the interior of the parameter box but non-smooth
  at boundaries (degenerate states, event-timing kinks). NLopt's
  own docs recommend Sbplx for "noisy or otherwise non-smooth"
  objectives.
- **BOBYQA** is faster on truly smooth problems, but its
  quadratic trust region fails badly when smoothness assumptions
  break. Use it only when you have reason to believe the
  objective is smooth.
- **COBYLA** when active linear-inequality constraints matter.
- **ISRES** / **CRS2** for global "is this the basin?" passes;
  slow.

## What does NOT ship in v1

Out of scope for this docs note (and for gh#40):

- `camdl fit run --backend ode`. Fit-side integration is deferred
  per the proposal §"Scope" — multi-stage scout/refine and
  per-stage `[stages.X] backend` in fit.toml need their own
  design pass. The CLI flag namespace stays clean for that
  future work.
- Bayesian inference under ODE. v1 is MLE-only. The natural
  future path is "fit with NLopt, Laplace-approximate around the
  optimum"; the book chapter should flag this when introducing
  ODE-backed inference.
- Reactive interventions × deterministic likelihood. Reactive
  interventions in stochastic models fire at different times
  across particles; in ODE they fire at one deterministic time.
  The likelihood difference is non-trivial. Typhoid has no
  reactive interventions so this doesn't bind v1, but it's a
  flagged caveat for any model that does.
