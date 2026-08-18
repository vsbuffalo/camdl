# PGAS: interval-observation-aware ancestor weights (gh#607)

Status: proposed Issue: gh#607 (root cause + evidence), gh#608 (surfacing
companion)

## Problem

`csmc_as`'s ancestor-sampling step uses the Lindsten–Jordan–Schön weight

    w̃_j = w_{s−1}^j · f_θ(x'_s | x_{s−1}^j)          (LJS 2014, Eq. 17)

(`fill_ancestor_log_weights`, `pgas.rs:1305`). That weight is exact when the
observation at each time depends on the state at that time alone. camdl's
interval streams violate this: a weekly `neg_binomial` bin scores the flows
_accumulated over the bin_, so the per-bin accumulator is part of the state the
observation reads, and it straddles the splice point. The AS step rewrites the
reference's ancestry mid-bin without re-scoring the bin, so the spliced
trajectory's bin total is **never evaluated against the data** — neither at the
AS step (no obs term in w̃) nor at the bin-end weighting (step 5 scores each
particle's own accumulators; the clamped reference scores the _reference's_, not
the hybrid's).

Two consequences, both observed (gh#607):

1. a splice can remove a bin's only flow, making the assembled trajectory's obs
   density exactly 0 — the chain then genuinely occupies −inf states (3,623 of
   6,000 sweeps in the reported run) and NUTS freezes;
2. even when no −inf occurs, hybrids carry an **unscored bin** — a bias in the
   trajectory posterior for every interval-observation model.

## The corrected weight

At AS substep `s`, let `B` be the set of interval streams whose current bin is
open (bin end strictly after `s`). Selecting ancestor `j` assembles a history
whose open-bin total, per stream `b ∈ B`, is

    n_b(j) = bin_so_far_b(j) + ref_suffix_b(s)

where `bin_so_far_b(j)` folds particle `j`'s accumulation since the bin start
(`acc[j]` + the unfolded `cum_flows[j]`, both already maintained), and
`ref_suffix_b(s)` is the reference's flow contribution from `s` to the bin end —
**j-independent, precomputable per sweep** by a suffix scan over the reference's
substeps.

The corrected ancestor weight multiplies in each open bin's end-of-bin
observation density at the spliced total:

    log w̃_j = log w_{s−1}^j + log f_θ(x'_s | x_{s−1}^j)
             + Σ_{b ∈ B} log p(y_b | n_b(j), θ)

Terms that do not depend on `j` cancel in the categorical draw and are omitted:
the reference tail beyond each bin end, and any **pointwise** stream (prevalence
/ current-pop reads the counts at the obs time, which after the splice are the
reference's for every candidate). Only interval streams contribute, and only
their open bins.

Correctness argument (the part to check): the CSMC target at the AS step is
p(ancestry | x'_{s:T}, y). With interval observations the likelihood term for
the open bin factorizes over (candidate head, reference tail) only through the
bin total `n_b(j)`, so the candidate-dependent part of p(y | ancestry) is
exactly `Π_b p(y_b | n_b(j))` — the term added. Closed bins are fully inside
either the head (already in `w_{s−1}^j` via step-5 weighting and resampling) or
the tail (j-independent). This reduces to LJS exactly when every stream is
pointwise (`B = ∅`).

Degenerate case: all candidates −inf (e.g. the data demand flows no particle can
supply) falls back to the reference's own internally-consistent history — the
existing `None → j_ref` arm — which from a finite-density reference preserves
finiteness. Invariant gained: **from a finite-density reference, `csmc_as`
returns a finite-density trajectory.**

## Cost

Per AS substep: one obs log-density per candidate per open interval stream — O(N
· |B|) scalar density evals on top of the existing O(N) transition densities,
inside the already-parallel `par_iter`. The suffix scan is O(substeps ·
transitions) once per sweep. No RNG-order change for the free particles (the
correction only reweights the ancestor categorical), so paired-seed CRN coupling
of the forward pass is untouched — but the ancestor draw itself consumes the
same RNG with different probabilities, so **PGAS chains are not bit-compatible
across this change** (expected: it re-keys under the engine-version fold
anyway).

## Companion fixes (same arc, separately commit-able)

- **(b) Chain-start validation** — `run_pgas` already computes and discards the
  initial reference's complete-data loglik; a −inf start currently only warns.
  Under `use_nuts`, redraw within bounds up to K times, then hard-error naming
  the chain and the failing stream/bin. (The MH path keeps its gh#471 escape.)
- **(c) NUTS −inf escape** — when `init_log_p` is non-finite at a sweep, run the
  MH-within-Gibbs block for that sweep (its accept handles `log_alpha = +∞`) and
  skip the dual-averaging update, mirroring gh#471, so a chain that lands at
  −inf can leave instead of freezing while its step size collapses.
- **(d) Surfacing (gh#608)** — per-chain count of non-finite-loglik sweeps as a
  `severity: error` diagnostic; an explicit branch (with a warn) where all-−inf
  weights currently normalize to uniform and silently skip an impossible
  observation (`types.rs:472–486`); the per-chain table distinguishes "stuck at
  −inf" from "no data".

## Tests

- **T1 (red at HEAD)** — the invariant: small chain-binomial model, one weekly
  NB stream, positive counts every bin, near-extinction params; across seeds,
  every `csmc_as` return from a finite-density reference has finite
  `complete_data_loglik`. Mutation check: revert the correction term, confirm
  the pinned seed goes red.
- **T2 (red at HEAD)** — NUTS-path `run_pgas` from a probability-0 start must
  not record sweep 0 at −inf.
- **T3 (red at HEAD)** — from a −inf state with a feasible θ region, the chain
  reaches finite loglik within N sweeps.
- **T4 (statistical, the settle-it test)** — on a 2-parameter model with an
  interval stream, PGAS posterior means/variances agree with a long PMMH run
  (which has no AS step and no splice) within Monte-Carlo error, before and
  after; before-the-fix disagreement on trajectory functionals (per-bin
  incidence) is the bias made visible.

## Out of scope

Re-deriving AS for `Exact` obs alignment beyond what the union grid already
gives; multi-cadence streams need no special casing (each stream's own bin end
defines its membership in `B`); the `mh` fallback path (no AS step, no splice).
