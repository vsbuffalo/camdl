# State-space PGBS spike: contract, measured feasibility, and the go/no-go experiment

Date: 2026-09-02\
Project: camdl\
Tags: pgas, pgbs, inference, state-space, design-spike

The design contract and de-risking measurements for an **opt-in, experimental**
state-space trajectory kernel (`csmc_bs`) beside the untouched innovation kernel
(`csmc_as`). Scope is the reviewer-agreed milestone: enough state-PGBS to pass
exact invariance on an enumerable toy and run the fixed-θ
`bvd_province_hier3_ksmooth` experiment at `sigma_se = 0`. Gamma overdispersion
and all performance optimization are deliberately out of scope until that
experiment reads out. Background: the innovation-representation diagnosis in
[`2026-09-01-hier3-ksmooth-pgas-profile.md`](2026-09-01-hier3-ksmooth-pgas-profile.md)
and the ancestor-sampling starvation instrument (PR#819).

## Why (one paragraph)

The conditioned object in today's PGAS is the innovation record (flows + gamma
noise); an ancestor change drags the reference's remaining innovations onto an
offset state path, and the exact suffix correction rejects ~97.5% of proposals
over long horizons. Measured: the cloud holds abundant locally compatible states
(finite fraction 0.61, admissible 0.24), AS converts almost none of it (early
renewal ≤ 0.026/sweep at N = 19,200; 0.005 at 2,400), and the early 70–80% of
the latent path effectively never renews. A kernel that conditions on the
**state path** and re-draws innovations locally removes the long-horizon
coupling by construction. Whether it converts the measured diversity into early
renewal is the experiment.

## The hard invariant: partially-collapsed Gibbs ordering

The joint target is π(θ, Z, U | Y) with U = (F, G) the per-substep innovations.
The sweep is, in this order and no other:

1. `Z′ ~ p(Z | θ, Y)` — the state kernel, innovations marginalized;
2. `U′ ~ p(U | Z′, θ, Y)` — edge-local reconstruction;
3. `θ′ ~ p(θ | Z′, U′, Y)` — today's complete-data NUTS, **unchanged**.

Step 3 must never see innovations from any earlier trajectory: collapsing U out
of step 1 and then conditioning on a stale U is not a Gibbs sweep and silently
targets the wrong distribution. The API must make the wrong ordering hard to
express — the state kernel returns only a state path; the complete
`PGASTrajectory` handed to the θ-move is constructible only through the
reconstruction seam. This invariant gets a test, not a comment.

## The Markov state

`Z_t = (X_t, A_t)`: integer compartment counts plus the open interval-stream
accumulators. A is not a nuisance: each accumulator adds a linear constraint
`ΔA = H·F` on the substep's flows, shrinking the flow-ambiguity lattice the
transition density marginalizes over. Sufficiency test (to be asserted in the
Phase-2 production note, stated here for the record): two histories with equal
`Z_t` must induce identical laws over all future latent states and observations.
For the `sigma_se = 0` prototype class — no persistent noise, no
reactive-intervention agenda — `(X, A)` suffices.

## The transition density, and what was measured about it

Per substep, `p(Z′ | Z)` = the sum of the innovation-conditional density over
integer flow vectors consistent with `(ΔX = S·F, ΔA = H·F)`. The classification
is **computed from the compiled IR, never hand-derived**:

1. collapse identical stoichiometry columns (their split marginalizes exactly
   into the group total — for hier3, infection/importation per province);
2. form `[S; H]` from the stoichiometry and the interval-stream projections;
3. integer nullspace ⇒ ambiguity directions;
4. enumerate the bounded lattice (non-negativity + group support) and sum.

Measured on hier3 (exact rational elimination over the IR): after step 1 the
nullspace is **2 dimensions per province, provinces uncoupled** — the
onset/confirm/community-death/facility-death diamond and the care-exit/ abscond
diamond. Lattice cardinality over 3,180 real posterior-trajectory edges (E1
runs, both N):

| statistic | terms per province-edge |
| --------- | ----------------------- |
| median    | 5                       |
| mean      | 38.6                    |
| p95       | 200                     |
| max       | 525                     |

The density factorizes by province as a **product of per-province sums** —
`log p(Z'|Z) = Σ_p log [ Σ_{F_p ∈ 𝓕_p} p(F_p | Z_p) ]` — so the enumeration cost
is the SUM of the three per-province lattice sizes; the Cartesian product across
provinces is never enumerated (and `Σ_p Σ_{F_p}` alone is not the probability —
the inner sums are combined in log space). Implied prototype cost at N = 2,400,
T = 104, all-N backward weights: ~1.4×10⁸ group-pmf evaluations per sweep —
seconds per sweep, acceptable for the experiment without any optimization. The
heavy tail is a **production** concern with a known path (the two directions
couple only through `confirm_die + m₂ − m₃ ≥ 0`, admitting a DP/prefix reduction
of the 2-D sum); per the agreed guardrails, none of that is built until the
experiment justifies it.

**Backward-candidate edges are not realized edges (measured).** The backward
weight evaluates `p(Z_{s+1}^chosen | Z_s^j)` for every candidate `j` — deltas
that never occurred in any forward simulation. Proxy measurement over cross-run
candidate pairs (6 independent N = 2,400 runs, counts-only — exact for
cardinality on this model since `H` annihilates both diamond directions; an
upper bound on feasibility): **75.2% of 9,450 candidate province-edges are
zero-compatible**, detected by the rational consistency check alone at
negligible cost; among the feasible remainder the lattice is **median 28, mean
391, p95 2,376, max ~9,000** — wider than realized edges because the candidate's
state mismatch is absorbed into the flows. Naive implied cost at N = 2,400:
~7×10⁷ terms/sweep, minutes-per-sweep — tolerable for the go/no-go run,
uncomfortable beyond it. Both proxy biases point down (accumulator constraints
only add zeros; within-cloud candidates concentrate tighter than cross-seed
draws), so `csmc_bs` must carry an in-kernel version of this instrument
(feasible fraction, lattice-size distribution, terms per sweep) and its numbers
supersede this proxy before C is interpreted. One free-exactness optimization is
in prototype scope: propensities depend only on `Z_s^j`, so they are evaluated
once per candidate and shared across its lattice terms.

Two testability requirements carried from review: (a) `H` must be generated from
the compiled observation/accumulator semantics (stream-, interval- and
missingness-aware), not extracted ad hoc — the extraction is part of the kernel
and gets its own oracle; (b) claims of the form "stream X cannot change the
nullspace" are asserted by test, not comment. (The first ad-hoc extraction in
this spike missed the exits streams' projection encoding — harmless here only
because `discharge` lies in no null direction, which is now exactly the kind of
fact the test must pin.)

## The kernel (deliverable B)

Backward simulation over stored particle states. The FREE particles reuse
today's machinery unchanged — propagation, resampling, observation scoring,
history storage, RNG streams: simulating `G → F → Z'` and keeping only `Z'`
already samples the marginal state transition. The REFERENCE slot does not: it
must be **pinned to the reference state path** — `Z_t = Z_t^⋆` at every substep,
accumulators included — never reconstructed by replaying the old innovation
record from an ancestor state. That pinning IS the representation change; an
innovation-conditioned reference inside something named `csmc_bs` would
reproduce the old pathology under a new name, so the state-conditioned reference
gets its own test (below). Then draw the final state from the final weights;
then for s = T−1…0 draw particle j with weight
`w_s^j · p(Z_{s+1}^chosen | Z_s^j)` over all N candidates (naïve all-N, per
guardrail — no subsampling in the prototype). Reconstruction then draws, per
stitched edge, the flows from the lattice-restricted conditional
`p(F | Z, Z', θ) ∝ p(F | Z, θ)` on the compatible set — the SAME lattice
enumeration and weights the backward density already computed, so reconstruction
is a categorical re-read of the density's own terms plus the merged-class split
conditionals, not a separate algorithm — yielding a complete `PGASTrajectory`
for the θ-move and outputs. (With gamma, the conditional becomes
`p(F, G | Z, Z', θ)`; same principle, harder integrand — deferred.)

Opt-in surface, following the `binomial`/`ancestor_sampling` identity pattern
verbatim: `trajectory_representation = "innovation" | "state"` and
`trajectory_kernel = "ancestor_sampling" | "backward"` on `Stage::PGAS`,
absent-means-today permanently, `skip_serializing_if` keeps default payloads
byte-identical, CLI overrides, unsupported combinations refuse loudly. `csmc_as`
is not modified.

## Gates before the experiment counts

- **Transition-density oracle**: `p(Z′|Z)` against brute-force enumeration on
  small models, and against forward-simulation frequencies.
- **Exact invariance, unique-flow toy** (nullspace-zero SIR, pop 5–10, T 3–5):
  initialize from the enumerated posterior, one kernel application, verify the
  posterior is preserved (`csmc_exact_invariance` style).
- **Exact invariance, ambiguous-flow toy**: a tiny enumerable model with a
  nonzero nullspace (a diamond A→B→D / A→C→D, ideally plus one accumulator
  constraint), so the invariance test exercises the COLLAPSED sum `Σ_F p(F|Z)` —
  the novel object — not just unique inversion. Required before production
  results are trusted.
- **State-conditioned reference test**: the reference slot holds `Z^⋆` itself at
  every substep (equality asserted against the retained path), and no code path
  applies a recorded flow delta to an ancestor state inside `csmc_bs`.
- **Reconstruction consistency**: reconstructed records satisfy the same
  complete-data density their edge conditionals imply.
- **Ordering invariant test** (see above).

## The go/no-go experiment (deliverable C)

Fixed θ (the fit config's curated start), hier3 with `sigma_se = 0` fixed. Arms:
PF-only (`ancestor_sampling = false`), innovation-PGAS, state-PGBS. **Same N
compared within N** — the innovation baselines are N-dependent (early renewal
0.005/sweep at N = 2,400, 0.026 at 19,200), so no cross-N thresholds. Start at N
= 2,400; primary outputs on **two axes, both recorded**: early renewal per sweep
(the statistical mechanism — did the representation change convert the cloud's
diversity?) and early renewal per CPU-second (the practical value). Both are
needed to read an outcome: 20× renewal/sweep at 5× CPU is a clear win; 2× at 10×
CPU is a kill; 50× at 20× CPU says optimize before deciding. If clearly
promising, repeat at 4,800/9,600 and then 19,200 for the envelope over N; rerun
at posterior-typical θ once chain viability lands (particle geometry is
θ-dependent). Kill criterion: if state-PGBS does not decisively beat the same-N
innovation baseline on the joint reading of those two axes, stop — gamma is
never built and `csmc_as` stands.

## Deferred, with reasons recorded

- **Gamma overdispersion**: postponed through C. Note for Phase D: the
  binomial-with-`1−e^{−aG}` gamma marginal has a closed form via the alternating
  Laplace-transform sum (external review, 2026-09-02) — the earlier "quadrature
  or augmentation only" claim was too pessimistic — but the alternating sum
  cancels catastrophically at large k, so production needs a stable-evaluation
  design (hybrid closed-form/quadrature); a separate problem from the
  representation question by construction.
- **M ≪ N subsampled backward weights**: production-relevant (the tail above
  says so), not prototype-relevant.
- **Full-DSL generalization**: the landing bar is feature parity with the
  innovation kernel — nothing ships for one model. The general fallback that
  makes parity a theorem rather than a hope (edge-local augmentation with joint
  rejuvenation, exact for arbitrary stoichiometry/noise, preserving locality) is
  recorded as the Phase-D design spine; the fast marginal path above is its
  optimization where the IR algebra allows.

## Result (2026-09-02, same day): the go/no-go tripped its kill criterion — with the mechanism identified

The three-arm experiment ran on `bvd_province_hier3_ksmooth_sigma0` (the
overdispersion wrapper removed at the model level; all 48 transitions plain),
fixed θ at the curated starts, N = 2,400, 30 sweeps, seed 51:

| arm             | early renewal (b0–b7) | aggregate | user CPU |
| --------------- | --------------------- | --------- | -------- |
| PF-only         | 0.056                 | 0.136     | 59 s     |
| innovation-PGAS | 0.091                 | 0.169     | 91 s     |
| state-PGBS      | 0.056                 | 0.136     | 215 s    |

The state arm's trace is byte-for-byte the PF-only arm's, at 3.7× the cost. The
in-kernel instrument says why, precisely: **exactly 1 feasible backward
candidate out of 2,400 at every level — the true genealogical parent.** The
backward stitch degenerates to retracing the forward genealogy.

The mechanism is structural, not a code defect (the kernel passes all three
exact-invariance gates, including the load-bearing-accumulator one): the `A`
component of `Z = (X, A)` remembers the CURRENT BIN'S ACTUAL FLOWS, so a
stitched edge from any other particle must reproduce the target's bin content
exactly from a different starting state and bin history. The `[S; H]` system's
consistency rows turn that into a set of exact integer constraints that, on a
9-stream model, essentially only the true parent satisfies. The very coordinate
that helped identifiability (ΔA shrinking the flow lattice) eliminates backward
diversity. The toys passed invariance because at population 5–10 exact
cross-lineage `A`-matches are common; at realistic scale they are measure-≈1/N.
Correctness and mixing came apart exactly where the theory says they can.

**Read against the spike's own criteria: stop.** No gamma work proceeds on this
formulation. `csmc_as` stands.

What survives, and the repair direction this measurement points at:

- The state-transition density module, its oracle suite, and the invariance
  harness are formulation-independent assets — any successor formulation reuses
  them.
- The repair is the one §12 of the downstream design note anticipated and this
  spike's `Z = (X, A)` choice tried to shortcut: keep `Z = X` only, and move
  interval observations into the transition factor by marginalizing the BIN'S
  flows jointly across its substeps (`p(y_bin, X-path segment)` via a joint
  lattice over the bin) — so cross-lineage stitching is constrained by states
  and observations, not by another particle's realized bin history. That is a
  substantially heavier derivation (multi-substep lattices, per-stream cadences,
  the leading-gap/never-closing-bin cases this model's exits streams exhibit),
  and whether it is worth attempting is a maintainer decision informed by this
  measurement — not an implementation default.
- The measured innovation-PGAS numbers above (AS worth 0.091 vs 0.056 early
  renewal at N = 2,400 on the sigma0 model — a larger AS contribution than the
  overdispersed model showed at this N) are a useful baseline regardless.

### The causal story, directly demonstrated (upstream-requested diagnostics)

Two diagnostic-only checks requested in review, run over 208 backward levels of
the C configuration (temporary instrumentation, not landed):

1. **The unique finite backward candidate IS the stored forward-resampling
   parent — 208/208 levels.** Asserted against the recorded ancestry, not
   inferred from the trace equality.
2. **Linear consistency with and without the accumulator rows:**

   | system                | consistent candidates per level (of 2,400) |
   | --------------------- | ------------------------------------------ |
   | `[S; H]` (Z = (X, A)) | mean 1.25 (min 1, max 11)                  |
   | `S` only (X-only)     | **2,400 — every candidate, every level**   |

   Without `H`, the counts system alone is consistent for the ENTIRE cloud (the
   collapsed S-nullspace has enough freedom to absorb any cross-particle counts
   delta; non-negativity and rate support would trim that toward the cross-run
   proxy's ~25% feasible, i.e. hundreds per level). With `H`, it collapses to
   essentially the parent alone.

So the closure is exact: **the accumulator rows, specifically, destroy the
backward connectivity** — from all 2,400 candidates to ~1. `Z = (X, A)` is a
dead end for mixing; `Z = X` with bin-marginalized observation factors remains
open, at materially higher derivation cost, as a separate authorization
decision.
