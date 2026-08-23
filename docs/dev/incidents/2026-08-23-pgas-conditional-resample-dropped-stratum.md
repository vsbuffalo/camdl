# The reference particle was denied its descendants for four months

- Date: 2026-08-23
- Issue: gh#718
- Fixed by: `55178dd1`
- Class: **code-vs-code** — the conditional resampling in `csmc_as` disagreed
  with the invariance requirement conditional SMC rests on. Fixed in code, with
  a test pinning the agreement.

## What happened

`csmc_as` — the sweep behind PGAS, camdl's default Bayesian backend — did not
leave the smoothing target `p(X | θ, y)` invariant. Every posterior it has
produced since `703945ba` (2026-04-05) is drawn from a distribution that is not
the one requested.

The kernel resampled by calling `systematic_resample`, which lays `n` evenly
spaced thresholds across the cumulative weight and returns one ancestor per
slot, and then used the result for the free slots only:

```rust
for j in 0..n_particles {
    if j == j_ref { new_counts.push(counts[j_ref].clone()); }
    else          { new_counts.push(counts[indices[j]].clone()); }
}
```

With `j_ref = n - 1`, `indices[n-1]` is never read. The thresholds are ordered
by cumulative weight, so discarding the last one confines the free slots to the
first `(n-1+u)/n` of the cumulative range. Every particle in the top `1/n` of
that range loses roughly one expected offspring — and the reference particle,
which sits last in the weight vector by construction, is always in it.

## Reproduction

The defect is visible in the resampler alone, with no model and no sweep:

```
$ cargo test --release -p sim --lib resampling
```

`conditional_systematic_matches_rejection_sampling_*` compares the draw against
rejection sampling from the unconditional cycle-randomised scheme. Against the
old scheme, the discrepancy is total rather than marginal. Measured directly on
a 5-particle ensemble, free-slot ancestor shares against the shares owed:

| `w_ref` | reference's fair share | actual | others |
| ------: | ---------------------: | -----: | -----: |
|    0.10 |                 0.1000 | 0.0000 |  ×1.25 |
|    0.40 |                 0.4000 | 0.2500 |  ×1.25 |

The deficit is about one slot regardless of `n`, and a typical particle only
expects about one slot, so it does not wash out with particle count: the
reference loses 59% of its fair share at `n = 5` and 65% at `n = 500`.

End to end, on a fixture whose entire trajectory support is enumerable, so the
target carries no Monte-Carlo error at all:

```
$ CSMC_INVARIANCE_M=400000 cargo test --release -p sim \
    --test csmc_exact_invariance -- --nocapture
```

Goodness-of-fit of one sweep's output against the exact `π`, before and after:

| ancestor sampling          | before | after |
| -------------------------- | -----: | ----: |
| off (plain particle Gibbs) |   6.52 | −0.11 |
| on                         |  10.76 |  1.16 |

## Root cause

Conditional SMC needs the `n-1` free ancestors drawn from the _conditional_ law
of a resampling scheme that is **marginally unbiased** — the law of a single
ancestor must be the weight vector itself (Chopin & Singh 2015, §5). Taking an
unconditional scheme and overwriting one slot is not that draw, and for a
stratified scheme it is not even close, because the discarded slot is not a
random one: it is the last stratum.

The correct construction is their Algorithm 4 (§5.2): condition `U` so the
reference receives an offspring with the right conditional law for how many, run
plain systematic selection at that `U`, then cycle the output uniformly over the
reference's own copies. `conditional_systematic_resample` implements it,
rotating the weight vector so the reference is first and rotating back.

Reference: Chopin, N. and Singh, S.S. (2015). "On particle Gibbs sampling."
_Bernoulli_ **21**(3):1855–1883, DOI 10.3150/14-BEJ629.

## Why the suite did not see it

Every test on this kernel pinned a _piece_ of it. The splice-ratio oracle pins
`splice_log_ratio` against `complete_data_loglik`; `pgas_ancestor_weight` pins
the Eq.-(17) weight; `csmc_splice_continuity` pins that the returned path does
not jump; `csmc_prevalence_only_invariance` pins a digest. Not one asked whether
a sweep applied to a draw from the target returns a draw from the target — and
the resampling had no test of its own beyond three assertions about the
_unconditional_ scheme, none of which touch the conditional case.

The obstacle was believed to be the ground truth. It was not. The support of a
non-overdispersed chain-binomial model over a few substeps at a small population
is finite and small; enumerating it gives `π` exactly, and the invariance test
becomes a multinomial goodness-of-fit with no nuisance parameters.

## What it cost, beyond the wrong answers

It sent a multi-day investigation to the wrong surface. gh#718 measured the
non-invariance correctly and attributed it to the gh#607 ancestor-sampling
splice, because it used **plain particle Gibbs as its provably-correct
control**. That control carried the same defect, so the paired `shipped − PG`
contrast subtracted one biased kernel from a more biased one and read the
residual as a splice defect. Ancestor sampling does amplify this bias — χ²/df
1.223 with it off, 1.384 with it on — which is exactly the signature a splice
defect would leave.

The splice is fine. With the resampling fixed it passes at z = 0.39 at 3× the
power of the run that indicted it.

## What this changes

1. **A control is only a control if it is tested.** "Provably correct" was true
   of plain particle Gibbs as an algorithm and false of this implementation of
   it. A control arm that shares code with the arm under test cancels shared
   defects out of the contrast and silently relocates them into the difference.
   Where a control is load-bearing, it needs its own absolute check, not just
   its role in a paired comparison.
2. **Test the kernel's defining property, not its parts.** Four tests pinned
   pieces of `csmc_as` and all four passed on a kernel that was not invariant.
   The invariance test now in the suite would have failed on day one.
3. **Reach for the exact fixture before the expensive one.** The instrument that
   settled this in minutes is a 6-individual SIR over 4 substeps. The one that
   could not settle it in days was a 15-substep model needing 800k
   importance-sampling draws for a ground truth. When a property is exactly
   checkable on a small enough state space, shrink the model rather than growing
   the sample.
4. **A parallel seam wants a named function.** The conditional draw was spelled
   as "call the unconditional one and ignore part of the answer," inline, with
   no name and no test. It is a different question from unconditional resampling
   and now has a function that says so.
