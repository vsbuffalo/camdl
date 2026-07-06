# PGAS ancestor-sampling weight dropped the importance weight

- Date: 2026-07-05
- Class: code bug (silent-wrong), inference math
- Affects: PGAS (`csmc_as`) — the default Bayesian inference method
- Fix: `rust/crates/sim/src/inference/pgas.rs` (`fill_ancestor_log_weights`)
- Reproduction: `rust/crates/sim/tests/pgas_ancestor_weight.rs`

## Summary

The conditional-SMC-with-ancestor-sampling (CSMC-AS) step scored the reference
particle's ancestor draw on the transition density alone,
`f_θ(x'_s | x_{s-1}^j)`, dropping the previous-substep importance weight
`w_{s-1}^j`. The correct ancestor-sampling weight (Lindsten, Jordan & Schön
2014, Eq. 3 / Eq. 17) is the **product**

```
w̃_j  ∝  w_{s-1}^j · f_θ(x'_s | x_{s-1}^j).
```

With the weight dropped, the reference lineage was spliced onto ancestors chosen
for transition plausibility while ignoring how well each ancestor explained the
most recent observation. This biases the ancestor draw at every substep whose
incoming weights are non-uniform — the substep following each observation — and
forfeits the Theorem-1 invariance of the PGAS kernel, so the sampler no longer
provably targets the exact joint posterior `p_θ(x_{1:T}, θ | y_{1:T})`.

## The authority

Lindsten, F., Jordan, M. I., & Schön, T. B. (2014). _Particle Gibbs with
Ancestor Sampling._ Journal of Machine Learning Research 15:2145–2184.

- Eq. (3), §3.2, p. 2150 (general):
  `w̃^i_{t-1|T} = w^i_{t-1} · γ_{θ,T}((x^i_{1:t-1}, x'_{t:T})) / γ_{θ,t-1}(x^i_{1:t-1})`.
- Eq. (17), §5.1, p. 2159 (state-space specialization):
  `w̃^i_{t-1|T} = w^i_{t-1} · p_θ(x'_{t:T}, y_{t:T} | x^i_{t-1}) ∝ w^i_{t-1} · f_θ(x'_t | x^i_{t-1})`.

The prose immediately below Eq. (17) is explicit that this is a product of two
factors: "the importance weight `w^i_{t-1}` is the prior probability of the
particle `x^i_{t-1}` and the factor `f_θ(x'_t | x^i_{t-1})` is the likelihood of
moving from `x^i_{t-1}` to `x'_t`. The product of these two factors is thus
proportional to the posterior probability that `x'_t` originated from
`x^i_{t-1}`." The categorical is over the **pre-resample** ensemble
`{x^i_{t-1}, w^i_{t-1}}`.

## Root cause: two audits collided into a frame-inconsistent hybrid

There are two self-consistent ways to draw the reference's ancestor:

1. **post-resample states + uniform weights** — score `f` alone over the
   resampled cloud; the importance weight is folded into resampling
   multiplicity.
2. **pre-resample states + `w·f`** — the exact Eq. (17).

The code ended up as neither:

- An earlier audit (IM6, 2026-04-19 inference review) dropped `log_weights` from
  the ancestor sum. That is correct **only** under formulation (1) — its comment
  argued "after resampling, all slots carry uniform weight 1/N."
- A later audit (`gh#audit-H8`) moved the ancestor **states** to the
  pre-resample ensemble (`prev_counts_for_ancestor[j]`, captured before the
  step-1 resample) to fix a separate state/weight _pairing_ mismatch — but left
  IM6's weight-drop in place.

Moving to pre-resample states (formulation 2) makes the pre-resample weight
`log_weights[j]` mandatory, yet it stayed dropped. The reference particle is not
resampled at step 1 (its slot is clamped), so the resample cannot inject the
weight either — the ancestor choice is driven solely by `ancestor_log_w`. The
block's own header comment already stated the correct formula
(`ã_j = w_{s-1}^j + log f(…)`) while the implementation computed `f` alone.

## How it was detected

Adversarial subagent review of the inference stack flagged the missing term.
Before treating it as real it was confirmed five independent ways: the paper's
Eq. (17); a direct code trace; and a three-agent adversarial panel — a neutral
judge, a prosecutor, and a **defender explicitly tasked to prove the code
correct** — all of which ruled BUGGY (the defender's steelman collapsed: the
"post-resample uniform" defense describes formulation 1, but the code executes
formulation 2's states).

## Impact

Bias, not merely degraded mixing. The AS step is one of the partially-collapsed
Gibbs steps whose exact form Theorem 1's invariance proof requires; using `f`
alone samples the ancestor index from the wrong conditional. It does **not**
degrade to valid Particle Gibbs (which would set `a^N = N`) — it is a third,
non-invariant kernel.

Magnitude scales with the **dispersion of the particle weights at observation
times** (≈ `1 − ESS/N`), because the dropped factor cancels in the softmax to
the extent it is constant across ancestors:

- near-uniform weights (noisy/uninformative observations, high ESS) → near-zero
  bias;
- dispersed weights (sharp observations, weak process noise, heterogeneous or
  spatial states, low ESS) → substantial bias.

Exposure also scales with cadence: at `dt` = observation spacing the bug fires
every substep; with finer `dt` only the first substep of each interval. The
direction is toward transition-plausible-but-data-inconsistent histories —
reconstructed latent trajectories biased toward smoothness around observation
times — which most plausibly biases process-noise / overdispersion parameters
downward (model-dependent).

## Reproduction (red → green)

`rust/crates/sim/tests/pgas_ancestor_weight.rs` builds a deterministic reference
via `simulate_reference`, computes the two candidate transition densities `td`
independently through the public `log_transition_density_substep`, and asserts
`fill_ancestor_log_weights` yields `log_weights[j] + td_j` under a non-uniform
`log_weights`.

Against the buggy formula (`*slot = td`):

```
test ancestor_weight_includes_importance_weight ... FAILED
free-slot weight = -5.882246332237088 but Eq(17) requires log_w + td = -6.182246332237088
  (the bug drops the log_w = -0.3 importance-weight term)
test result: FAILED. 0 passed; 1 failed
```

The −0.3 gap is exactly the dropped importance weight. With the fix
(`*slot = log_weights[j] + td`) the test passes.

## Fix

`*slot = log_weights[j] + td;` at the single ancestor-weight assignment. The
computation was extracted from the inline `csmc_as` loop into a documented,
`pub` `fill_ancestor_log_weights` so the exact quantity Theorem 1 depends on is
unit-testable in isolation — it had been buried in a 400-line sweep, which is
how the term went missing behind two layers of individually-plausible comments.

## What it changes going forward

- **Extract load-bearing formulas into testable units.** The bug survived two
  audits because the weight lived in an untestable inline closure. The named
  `pub` helper + its regression test make the Eq.-(17) form pinned and
  greppable.
- **When two fixes touch the same code, re-derive the _combined_ invariant.**
  IM6 and H8 were each locally defensible; their composition was not. A comment
  that justifies a step in terms of a frame a later change invalidates is a
  latent bug.
- **Verify inference math against the primary source, not internalized
  understanding.** The confirmation that mattered was reading Eq. (17), not
  recalling "the PGAS-AS weight."
