# PGAS returned trajectories outside the target's support, and silently dropped them from output

Date: 2026-08-19 Issue: gh#607 Fixed: 4dbe88ac..5b04dba4
(`camdl 0.1.0+5b04dba4`) Severity: incorrect inference on every PGAS fit;
user-visible data loss

## What happened

`csmc_as`'s traceback stitches per-substep records along an ancestry that
ancestor sampling reassigns, but the reference slot's recorded `counts_before`
was always its OWN, never that of the ancestor it had just been assigned. The
returned trajectory therefore JUMPED in compartment state at every accepted
splice, and `complete_data_loglik` — which reads each record's stored
`counts_before` without checking continuity — assigned that discontinuous path a
finite density.

Every path in the smoothing target's support satisfies
`counts_after(s−1) == counts_before(s)`. A kernel that returns points outside
the support cannot be invariant for the target, so the number the θ-step
conditioned on was not the model's likelihood of anything.

Separately, the ancestor was drawn from the Markov weight (LJS Eq. 17) in a
model that is non-Markov in the recorded state: with interval observations the
per-particle flow accumulator is part of the extended state and entered neither
the weight nor the post-splice bookkeeping.

## How it was detected

Not by a test — by a user. The downstream ebola-bdbv project reported a chain
holding `log_posterior = −inf` as its CURRENT state for 3,623 of 6,000 sweeps
with 0.37% acceptance, and asked whether the sampler could occupy a zero-density
state or whether the trace was recording the wrong thing.

Trace forensics on the retained run answered it: all 3,623 −inf rows carried
`obs_ll = −inf` with FINITE `transition_ll` (so the zero density was in the
observation term), mean `trajectory_renewal` 0.948 with renewal > 0 on every
such row (so CSMC was delivering FRESH trajectories that then scored −inf — the
splice, not degenerate retention), and `tree_depth > 0` throughout.

A direct test then settled the scope: 4 discontinuities across 8 CSMC sweeps on
a plain SIR. **Not an interval-observation bug — every PGAS fit.**

## Why it survived

- `complete_data_loglik` never checked state continuity, and the traceback's own
  `debug_assert` block checked time contiguity and per-record density only — and
  is compiled out in release.
- The −inf symptom looked like a _user_ problem (bad starts, bad priors), and
  the existing diagnostics described it that way: the degenerate-AS warning said
  "reference trajectory is too far from particle cloud. Consider more particles
  or smaller parameter proposals."
- Ancestor sampling was tested for its WEIGHT (`pgas_ancestor_weight.rs` pins
  Eq. 17) but never for the OBJECT it returns.
- Parameter-recovery studies passed, which is consistent: splices are rare, each
  jump is small because the transition term already favours compatible
  ancestors, and the bias is small in well-identified regimes.

## The fix

1. The splice is applied at source — the reference slot's pre-state becomes the
   sampled ancestor's end-state and the recorded net delta is applied — so
   continuity holds for every particle by construction. Feasibility of the whole
   shifted suffix reduces to a per-compartment headroom array (one backward
   pass), because a splice holds the recorded noise fixed.
2. The ancestor draw became a Metropolis move: the cheap Eq.-17 weight is the
   proposal, accept/reject uses the exact ratio (LJS §6.1–6.2, Eq. 21), which
   restores exactness while KEEPING ancestor sampling — plain particle Gibbs
   would have been provably correct but degenerate on long series.
3. A state that cannot consume exactly the recorded noise multipliers now scores
   zero density, converting a silent positional re-pairing into a rejection.
4. The interval accumulator re-syncs on reassignment, from snapshots taken in
   the pre-resample index space.

Guarded, not assumed: the constant-offset argument fails for `events {}`,
scheduled compartment interventions, and `balance {}` rewrites, which recompute
counts from state. Those splices are refused (a mixing cost, not a correctness
one).

## Measured impact

|                                                                 | before  | after                            |
| --------------------------------------------------------------- | ------- | -------------------------------- |
| −inf draws in the RETAINED posterior                            | 510     | **0**                            |
| non-finite sweeps (4 chains × 1200)                             | 880     | 143 (all pre-burn-in, one chain) |
| a frozen chain's θ acceptance                                   | 0%      | 72%                              |
| R̂ (β)                                                           | 10.55   | 2.72                             |
| posterior trajectories silently missing from `trajectories.tsv` | **218** | **0**                            |

Posterior shift among chains that mixed under the defect: β +2.8%, σ +2.7%, γ
+5.5%. Pooled including the previously-frozen chain: β −15.4%.

Cost: 0.95×–1.13× in `csmc_as`, flat in T — the feasibility walk early-exits at
the first impossible term, so it is O(T · ~7) typical against O(T²) worst case.
**Mixing: the in-repo figure UNDERSTATES the cost on a real model, and a budget
planned from it will be short.** The synthetic measurement was trajectory
renewal 0.99 → 0.90 (0.80 with a scheduled intervention). The downstream ebola
project measured **0.884 → 0.605** on their model family at 40,000 sweeps, and
it shows up where it matters — worst-parameter R̂ went from 1.22 pre-fix to 1.91
(`tau`) and 2.04 (`q_comm`) post-fix **at the same budget**.

**On those R̂ figures, corrected twice and now settled by measurement.** They
were computed on the SEVEN sampling chains only — the chain seeded at `-inf` was
dropped by the downstream loader — so they are NOT inflated by it. (This
incident previously carried a caveat claiming otherwise; the downstream team
recomputed both ways and refuted it.) The like-for-like comparison stands:
`labruns_full` had eight finite chains with worst-parameter R̂ 1.22; this run has
seven finite chains with `q_comm` 2.04 and `tau` 1.91. Two caveats they raised
and worth keeping: the runs differ in chain count (7 vs 8) and R̂ is mildly
chain-count sensitive, and equal sweeps is not equal effective samples — which
is the whole point of the renewal figure.

**The general hazard is real and their data demonstrates it**, separately from
this incident: pooling the degenerate chain moves `rho` 1.08 → 1.43 and `tau`
1.91 → 2.17. And `camdl fit summary` reported **max R̂ 9.497** on the pooled
cloud for this run — dominated by a chain the same summary flags as degenerate.
Both figures are correct for what they are, but the actionable one requires the
reader to recompute. Filed separately. The old renewal figure was inflated by
counting splices that were off the target's support, so this is an honest cost
rather than a regression: the sampler now declines moves it should never have
made, and needs more sweeps to cover the same ground. Anyone sizing a run should
plan from the field number, not the synthetic one.

The trajectory loss deserves its own line: `coherent_counts_after` refused
discontinuous paths on its negative-count guard, so the splice defect surfaced
as **output silently missing 218 of the saved draws** — on the artifact
`simulate --init-state` reads as its forecast source.

## What changes because of this

- **`csmc_splice_continuity.rs`** asserts the returned trajectory is a
  continuous model path. It was committed RED and `#[ignore]`d before the fix
  existed, then un-ignored — the test that proved the bug proves the fix.
- **`csmc_splice_ratio_oracle.rs`** pins the acceptance ratio against
  `complete_data_loglik` non-circularly: it derives the trajectory and asks the
  likelihood, rather than re-deriving the ratio.
- A proposal's mathematics is not verified by its author. The first fix proposed
  here was REFUTED by a review against the primary sources (its weight truncated
  the future path, and it relied on a cancellation that fails because camdl's
  innovations are state-dependent where the source's are not). The second fix
  was ALSO returned once, for claiming a gamma cancellation that a state-gated
  term set does not permit.
- A green test suite established no regression, never correctness. Two tests
  existed over this code path and neither could detect a wrong acceptance ratio.

## Follow-ups

- ~~Chain-start validation is still warn-and-continue: a chain seeded in a bad
  basin runs to completion. The residual 143 non-finite sweeps are this.~~ Done
  (gh#607): a chain whose complete-data log-posterior is non-finite at its start
  AND still non-finite after its first Gibbs sweep is refused, skipped with a
  `BadInit` diagnostic, and excluded from `draws.tsv`, R̂ and every pooled
  number; the run errors only if every chain is refused. The second half of the
  test matters — the `X|θ,y` move rescues an unlucky reference draw at the same
  θ, and does so in three of this repository's own PGAS fixtures.
- The NUTS `−inf` absorption has no escape analogue to gh#471.
- gh#658 / gh#659: a chain can also stop sampling mid-run when its frozen step
  size no longer suits the region it wandered into — a different failure with a
  different signature, found the same night.
