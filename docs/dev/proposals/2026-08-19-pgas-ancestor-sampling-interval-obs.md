# PGAS ancestor sampling is not target-invariant for interval observations

Status: proposed Issue: gh#607 Scope: `rust/crates/sim/src/inference/pgas.rs` —
the highest-risk surface in the repo. Correctness of every interval-observation
PGAS posterior.

## The defect

CSMC with ancestor sampling draws the reference particle's ancestor with
(`fill_ancestor_log_weights`, pgas.rs:1304-1345):

```
log w̃_j = log w_{s-1}^j + log f_θ(x'_s | x_{s-1}^j)         (LJS 2014, Eq. 17)
```

That is the MARKOV weight. With interval observations the recorded compartment
state is not the whole state: a stream that scores accumulated flow over
`(previous obs, this obs]` carries a per-particle accumulator (`cum_flows`,
folded into `acc`) that is part of the extended state, and the model is
non-Markov in the recorded state (LJS 2014 handles this case separately). Two
concrete consequences in the code:

1. **The accumulator is absent from the weight.** Mid-interval every
   `log_weights[j]` is uniform 0 (pgas.rs:1667-1670, the non-observation
   branch), so the ancestor choice is made on one substep's transition density
   alone — blind to how much flow each candidate ancestor has already
   accumulated in this interval.
2. **The splice leaves the accumulator inconsistent.** When AS reassigns the
   reference's ancestor (pgas.rs:1624-1634), `cum_flows[j_ref]` and `acc[j_ref]`
   are NOT re-synced to the chosen ancestor's; the accumulation loop immediately
   after (pgas.rs:1636-1642) keeps adding to the reference's own totals. The
   traceback (pgas.rs:1697-1711) then stitches ancestor-prefix +
   reference-suffix flows, so the interval the selected trajectory actually
   carries is a hybrid **no filter weight ever scored**.

When that hybrid interval sums to zero flow against a positive observed count,
`complete_data_loglik` returns −∞ for EVERY θ — the chain's current state has
zero posterior density, NUTS is absorbing there (every leapfrog leaf diverges),
and the chain freezes. That is gh#607's visible symptom, but the invariance
break is present on every sweep, not only the −∞ ones.

**Confirmed on the retained ebola run** (`fit_national_base-7f86eaa7`, chain_4):
all 3,623 −∞ rows carry `obs_ll = −inf` with FINITE `transition_ll` (the zero
density is in the observation term, i.e. the accumulator, not the dynamics);
mean `trajectory_renewal` 0.948 with renewal

> 0 on every such row (CSMC delivered FRESH trajectories that then scored −∞ —
> the splice mechanism, not degenerate retention of a bad reference);
> `tree_depth > 0` throughout (NUTS, matching acceptance 0.37%).

## Fix

**Step 1 — re-sync the reference slot's interval state on every AS
reassignment.** Before the accumulation loop, when `ref_ancestor != j_ref`, copy
`cum_flows[ref_ancestor] → cum_flows[j_ref]` and
`acc[ref_ancestor] →
acc[j_ref]`. The reference's own `substep_flows` at `s`
stay (they are the noise AS conditions on); only the PREFIX moves, which is
exactly what the traceback will stitch. This makes the filter weight the
reference slot receives at the next observation equal the obs term of the
trajectory the traceback would return — so a splice that would score −∞ gets
weight −∞ at the next observation and is (almost surely) not selected. Cheap,
surgical, and it removes the −∞ entry path.

Step 1 alone does NOT restore invariance: the ancestor is still CHOSEN without
the accumulator.

**Step 2 — the non-Markov ancestor weight.** The reference trajectory is fixed
for the sweep, so the flows from `s` to the next observation are known. The
correct weight adds the due observation's log-likelihood evaluated at (candidate
ancestor's accumulated prefix + reference's suffix flows):

```
log w̃_j = log w_{s-1}^j + log f_θ(x'_s | x_{s-1}^j)
                        + log g_θ(y_next | acc_j ⊕ ref_suffix_flows)
```

which reduces to the current formula exactly when no interval stream is due (the
term is constant across `j`). Cost: one obs-likelihood evaluation per candidate
per substep — the same shape as the existing per-particle scoring, inside the
existing rayon fan-out.

**Step 2′ — conservative alternative if Step 2's cost or derivation does not
hold up under review:** restrict ancestor sampling to substeps where
`reset_due_acc` has just fired (interval boundaries). There the accumulator is
zero for every candidate, the extended state IS the compartment state, and Eq.
17 is exact. This provably restores invariance at the cost of fewer AS
opportunities (worse mixing, not wrong answers). Ship 1 + 2′ if 2 is not ready;
never ship 1 alone and call it fixed.

**Step 3 — guards, each independently valuable:**

- refuse (or resample) a non-finite initial complete-data log-likelihood instead
  of the current warn-and-continue (pgas.rs:2244-2283, 2352-2389);
- a release-mode finiteness check on `csmc_as`'s returned trajectory;
- per-chain `no_finite_anchor`, not global-best-only (mod.rs:80-82) — today one
  frozen chain in four exits 0 and pools (the gh#635/608 surfacing already flags
  it at read time; this refuses at write time).

## The invariant test (the deliverable that proves it)

On a two-interval interval-observation model, seeded so AS joins mid-interval
onto an ancestor whose prefix flows are zero while the data records cases:

> **the selecting filter weight is finite ⇒ the returned trajectory's
> `complete_data_loglik` is finite**, and more strongly, the obs term of the
> returned trajectory equals the weight that selected it.

Currently violated; this is the red→green. Plus: a regression that a −∞-forcing
start refuses rather than streaming −∞ trace rows, and a byte-identity check
that models with NO interval streams are unaffected (instant/prevalence
observations must produce identical trajectories and run_ids — the accumulator
term is constant across `j` there).

## Validation beyond the unit test

The unit test proves the mechanism; it does not quantify the bias the released
sampler carried. Before/after on the retained ebola fit (same seed, same data,
same config): the −∞ rows must vanish, and the posterior shift on γ/τ/R_eff is
the measured magnitude of the defect — a number we owe the downstream team, who
fitted under it. Simulation-based calibration on a small interval-obs model is
the stronger check if the shift is material.

## Risk

`pgas.rs` is the highest-risk surface named in CLAUDE.md; PGAS is the default
Bayesian backend for the downstream national fits. Steps 1 and 2 change sampled
trajectories, so goldens/traces move by construction — the byte- identity check
for non-interval models is what bounds the blast radius. Do not batch these with
unrelated changes; each step is its own commit with its own green gate.
