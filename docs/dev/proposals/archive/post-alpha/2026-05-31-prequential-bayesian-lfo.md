---
status: superseded 2026-08-29 by 2026-08-29-honest-predictive-evaluation.md
date: 2026-05-31
---

# Proposal: Bayesian leave-future-out prequential evaluation (LFO-PSIS)

**Status:** Proposal **Date:** 2026-05-31 **Split from:**
`archive/pre-alpha/2026-04-20-prequential-evaluation.md` (Part I — the plug-in
prequential layer: `camdl compare` log predictive density, `--save-prequential`,
the `PrequentialTrace` — shipped pre-alpha. This proposal carries only the
deferred Part II remainder so the live set describes open work; see the archived
original for the full framing, references, and the Part I/II split rationale.)
**Motivation:** Part I scores plug-in predictive density at a point estimate
(IF2/MLE, profile). The Bayesian extension — propagating parameter uncertainty
through the predictive — was deferred because every inference pipeline that
mattered for Part I's worked example was plug-in. It becomes load-bearing as
soon as we report predictive comparisons under a posterior (PGAS/PMMH).

## Scope (the deferred Part II)

The open work, carried verbatim from the original's Part II table:

- **Fully Bayesian predictive via LFO-PSIS** — approximate leave-future-out with
  Pareto-smoothed importance sampling (Bürkner, Gabry & Vehtari 2020), reusing
  the posterior draws rather than refitting per fold.
- **Pseudo-posterior provenance from the IF2 particle cloud** — define and
  record what "posterior" the LFO weights are taken against when the source is
  IF2, not a true posterior sampler.
- **PSIS k̂ diagnostic** — surface the Pareto-k tail-reliability per fold;
  gate/​warn when k̂ exceeds the usual thresholds (refit-needed signal).
- **Randomized PIT for discrete observations** — calibration check that is
  honest for count data (avoid the discrete-PIT staircase artifact).
- **Identifiability sweep for `t_0`** (opt-in, then default) — detect when the
  introduction time is not pinned by the predictive scoring.
- **Rolling-origin k-step-ahead** — the panel/forecast-horizon variant.
- **Energy score / panel comparison UX** — multivariate proper scoring + the
  surface for comparing across panels.

## Open design questions (to settle before implementation)

- Which inference sources qualify as a "posterior" for LFO (PGAS yes; IF2 cloud
  only with a stated pseudo-posterior contract).
- Where the k̂ gate lives (a diagnostic alongside the existing refine gates, or a
  `compare`-time warning).
- CLI surface: extend `camdl compare` vs a dedicated `camdl predict`/`lfo`.

This is design-only until those are settled; nothing here is implemented.
