# Incident: Spatial PGAS -inf complete-data log-likelihood

**Date:** 2026-04-07 **Status:** Resolved. Six bugs fixed across engine and CLI.
**Severity:** Blocking — spatial PGAS inference produced -inf on every sweep,
making Bayesian inference impossible for multi-patch models. **Fix commits:**

- `f64668f` — snapshot counts_before in SubstepRecord (simulate_reference)
- `b15cb39` — reference particle counts_before mismatch in CSMC-AS traceback
- `faffe8f` — near-zero rate/flow mismatch: soft handling + iota warning
- `9547ef9` — deterministic check ordering in density evaluator
- `c20de85` — IVP density uses per-patch population for spatial models
- `8a0e9f9` — multi-stream data loader reads correct column per stream

## Fundamental vs. implementation

This incident has two layers.

**The implementation bug** was: the density used post-clamp counts while
`step_one` used pre-clamp counts. This is purely a coding error — the density
and simulator must agree on what state generated the draws. Fixed by storing
both snapshots in `SubstepRecord`.

**The fundamental issue** is: the Euler-multinomial approximation assumes
per-step exit probabilities are small ($p_{\text{total}} \ll 1$). When R0 is
large and dt = 1 day, $p_{\text{total}} = 1 - e^{-R_0 \cdot \gamma \cdot
\Delta t}$ can approach 1, meaning almost everyone exits S in one step. With
spatial models, multiple source groups draw from overlapping compartments via
the deferred-delta mechanism, and total withdrawals can exceed the population —
the overdraft that triggers clamping.

~~This is a known limitation of the Euler scheme. pomp has the same issue and
handles it the same way (clamp negative counts).~~

**Correction (2026-10-26).** Both claims in the struck sentence are wrong. The
overdraft is not a property of the Euler scheme: an Euler-multinomial draw
splits one source's exits as Multinomial(N; p₀…p_K) and cannot exceed N. It
arose here from several source groups drawing on one compartment, which camdl
has rejected since gh#121. pomp (King, Nguyen & Ionides 2016, _J. Stat. Softw._
69(12); source checked at v6.4.0.6) does not clamp: `reulermultinom`
(`inst/include/pomp.h:157-189`) returns NA with the warning "NAs produced" for a
negative or non-integer size. Its particle filter flags a NaN or +∞ log-weight
(`src/pfilter.c:76-81` returns the offending particle's index to the caller) but
accepts −∞ for a particle, and gives the step a log-likelihood of −∞ when every
weight is −∞ (`:87-92`). The original sentence gave no citation.

The correct fix for the approximation is smaller dt (subdivide each day into
4–10 substeps), which keeps $p_{\text{total}} < 0.3$ and makes overdrafts
vanishingly rare. But smaller dt means more substeps, which means slower
inference.

The `counts_before`/`counts_after` fix addresses the implementation bug. The
Euler approximation breakdown requires either smaller dt or the
merge-same-destination optimization (see Remaining Issues).

## Summary

The complete-data log-likelihood (`complete_data_loglik`) returned `-inf` for
spatial (multi-patch) models during PGAS inference. The root cause was a
mismatch between the compartment counts used by `step_one` to draw transitions
and the counts stored in the trajectory for density evaluation. Negative
clamping in `step_one` reduced compartment counts after draws, so the density
evaluated `Binom(k; n, p)` with `k > n`, which is mathematically impossible and
returns `-inf`.

## Background

PGAS (Particle Gibbs with Ancestor Sampling) requires evaluating the
complete-data log-likelihood over a stored trajectory. This decomposes into:

```
log p(y, X | θ) = log p(x₀|θ) + Σ_s log p(x_s | x_{s-1}, θ) + Σ_k log p(y_k | x, θ)
```

The transition density `log p(x_s | x_{s-1}, θ)` uses the Euler-multinomial
decomposition: total exits from source compartment `j` are
`Binom(n_exit; N_j, 1 - exp(-R·dt))`, then split proportionally among competing
transitions. The constraint `n_exit ≤ N_j` is fundamental — binomial with
`k > n` is undefined.

Spatial models have inter-patch coupling via importation terms (e.g.,
`c_ij * I_j / N_j`). These coupling terms can create transient negative
compartments when outflows exceed the current count in a single substep. The
simulator handles this with "negative clamping" in `step_one`: after all draws,
any compartment that went negative is clamped to 0.

## The bug

The trajectory's `SubstepRecord` stored only the post-step compartment counts
(after clamping). When the density evaluator needed the counts _before_ a
substep, it read `trajectory.substeps[s-1].counts` — the _previous substep's
post-clamp_ state. But `step_one` had evaluated propensities and drawn exits
from the _pre-clamp_ state of the _current_ substep (which is the post-step
state before clamping was applied).

The mismatch:

1. At substep `s`, `step_one` sees compartment `I[p5] = 264` (pre-clamp)
2. It draws `n_exit = 264` exits (all individuals leave)
3. Negative clamping reduces `I[p5]` to 249 (because inflows from other patches
   partially compensated, but the net was still an overcount)
4. The trajectory stores `counts[s] = [..., 249, ...]`
5. At substep `s+1`, the density evaluator reads
   `counts_before = trajectory.substeps[s].counts = [..., 249, ...]`
6. It evaluates `Binom(264, 249, p)` — but `k=264 > n=249`, so the result is
   `-inf`

This never affected single-patch models because clamping only triggers with
spatial coupling (importation terms create negative transients). It also didn't
affect the particle filter, which evaluates likelihood at observation times
only, not per-substep.

## Investigation timeline

The investigation involved collaboration between the engine developer and a
downstream vignettes agent testing a 5-patch SEIR polio model.

### Phase 1: Reproduction and isolation (commits `4a6c2e3`–`d3a5c82`)

The downstream agent reported that PGAS produced `-inf` on their spatial model.
Initial diagnostics were added to `complete_data_loglik` to identify which
substep first produced `-inf`. A standalone spatial density test suite was
created (`tests/spatial_density.rs`) with SIR, SIR+demography, two-patch, and
5-patch models.

**Key finding:** All density tests passed at true parameter values. The `-inf`
only occurred during actual PGAS inference, not in the standalone tests. This
led to an incorrect hypothesis that the issue was parameter-specific.

### Phase 2: Model-specific vs. code-level (commits `d223f02`–`234d120`)

The downstream agent provided their compiled IR and exact parameter values.
Testing with their model at their parameters _also passed_ the round-trip
density test (100/100 seeds). This was confusing — the same model failed during
PGAS but passed in isolation.

**Key insight:** The difference was that the standalone test evaluated density
immediately after `simulate_reference` (using the same trajectory), while PGAS
evaluated density after a CSMC sweep had modified the trajectory. The CSMC
traceback reconstructed counts from history, and the history stored post-clamp
counts.

### Phase 3: Hypotheses explored and rejected

Several hypotheses were investigated:

1. **Gamma density indexing mismatch.** The gamma multiplier index tracking
   between `step_one` and `log_transition_density_substep` was found to be
   inconsistent. The gamma density was disabled (`if false {}`) but `-inf`
   persisted, ruling out gamma as the sole cause.

2. **Floating-point threshold divergence.** The zero-rate threshold in
   `step_one` (originally `0.0`) differed from the density evaluator (also `0.0`
   but with different floating-point behavior). Both were aligned to `1e-15`,
   but this was not the root cause.

3. **Merge-same-destination optimization.** A proposal from the upstream
   colleague suggested merging transitions with identical source AND destination
   stoichiometry to eliminate the multinomial split for spatial importation
   groups. This would be a correct architectural fix but was deferred — the
   snapshot fix was simpler and more direct.

### Phase 4: Root cause identified (commit `f2f614a`)

Enhanced diagnostics printed the exact `Binom(k, n, p)` arguments at the failing
substep. The output showed `Binom(264, 249, p)` — `k > n`. Once this was
visible, the cause was obvious: the density was using post-clamp counts as `n`,
but the flows were drawn from pre-clamp counts.

### Phase 5: Fix attempts

**Attempt 1 (commit `c57ffe6`):** Added `n_exit > n_src` guards and capping
logic. This was a bandaid — it prevented `-inf` but changed the statistical
model (capping `n` to `n_exit` gives `Binom(k; k, p) = p^k`, which is wrong).

**Attempt 2 (commit `f64668f`, final fix):** Store a pre-step snapshot
(`counts_before`) in `SubstepRecord` alongside the post-step state
(`counts_after`). The density evaluator reads `counts_before` directly — the
exact counts that `step_one` used when drawing exits. This is mathematically
correct: the density evaluates the probability of the observed flows given the
state that actually generated them.

### Phase 6: Second bug — CSMC reference counts_before (commit `b15cb39`)

After deploying the snapshot fix, the downstream agent rebuilt and still got
`-inf`: `Binom(677, 670, p)` at `src_comp_idx=3`. This was NOT a clamping issue
— it was a separate bug in `csmc_as`.

In CSMC-AS, each substep: (1) resamples particles, (2) saves
`prev_counts[j] = counts[j]`, (3) propagates free particles, (4) clamps the
reference particle to its stored trajectory. Step 2 saves `prev_counts[j_ref]`
from the post-resample state — which after resampling could be _any_ particle's
state, not the reference's actual pre-step state. But the reference's flows
(`ref_rec.flows`) at step 4 were drawn from `ref_rec.counts_before`.

The traceback paired `counts_before = prev_counts[j_ref]` (wrong) with
`flows = ref_rec.flows` (drawn from a different state), producing
`k > n → -inf`. Fix: after step 4, overwrite
`prev_counts[j_ref] = ref_rec.counts_before`.

## The fixes

### Fix 1: Snapshot in SubstepRecord (simulate_reference)

`SubstepRecord` was changed from:

```rust
pub struct SubstepRecord {
    pub counts: Vec<i64>,    // post-clamp
    pub flows: Vec<u64>,
    pub gammas: Vec<f64>,
}
```

to:

```rust
pub struct SubstepRecord {
    pub counts_before: Vec<i64>,  // pre-step snapshot (what step_one saw)
    pub counts_after: Vec<i64>,   // post-step, post-clamp
    pub flows: Vec<u64>,
    pub gammas: Vec<f64>,
}
```

All density evaluation paths (`complete_data_loglik`,
`complete_data_loglik_grad`, `log_transition_density_substep`) now use
`rec.counts_before` instead of deriving counts from the previous substep.

All trajectory construction paths (`simulate_reference`, CSMC-AS traceback) now
store both fields. The CSMC history tracks `history_counts_before` (the pre-step
particle states) and `history_counts_after` (post-step states).

**Files changed:** `pgas.rs`, `pgas_grad.rs`, `cli/fit/pgas.rs`,
`tests/pgas_resume.rs`, `tests/spatial_density.rs`

**Memory overhead:** One extra `Vec<i64>` per substep (n_compartments integers).
For a 5-patch SEIR model (20 compartments) with 1000 substeps, this adds ~160 KB
per trajectory — negligible compared to the particle array.

### Fix 2: Reference particle counts_before in CSMC-AS

After clamping the reference particle (step 3 in the CSMC loop), overwrite
`prev_counts[j_ref]` with the reference's actual pre-step state:

```rust
prev_counts[j_ref].copy_from_slice(&ref_rec.counts_before);
```

This ensures the history correctly pairs the reference's pre-step state with its
flows, regardless of what resampling did to the `j_ref` slot.

### Fix 3: Near-zero rate/flow mismatch (iota warning)

When `step_one` evaluates a spatial importation expression like
`c_ij * I_j / N_j * S_i`, floating-point arithmetic can produce a near-zero but
nonzero rate (e.g., `1e-16`) even when `I_j = 0`. If step_one draws `flow=1`
from this near-zero rate, the density evaluator may recompute the rate as
exactly zero (or below `RATE_EPSILON`) and reject the trajectory.

Before this fix, any `flow > 0` with `rate ≤ RATE_EPSILON` returned `-inf`.
After:

- **rate == 0.0 exactly, flow > 0:** Still `-inf`, but emits a one-time warning:
  "transition X has rate=0 but flow=N — consider adding a seeding term (iota)."
  This catches the model specification issue where infection rates vanish when a
  compartment empties.
- **0 < rate ≤ RATE_EPSILON, flow > 0:** Include the transition in the
  multinomial with its tiny rate. The Binomial density gives a very negative but
  finite score, correctly penalizing the unlikely event without hard-rejecting
  the trajectory.

User-facing documentation added to `docs/inference.md` under "Spatial models and
seeding (iota)".

### Fix 4: Deterministic check ordering in density (`9547ef9`)

The density combined `rate <= RATE_EPSILON || is_determ[tr_idx]` in one branch.
A deterministic transition with positive rate entered the near-zero handler
instead of being skipped from the multinomial. This could inflate `n_exit` by
including deterministic flows. Fixed by checking rate first, then deterministic
— matching step_one's exact order.

### Fix 5: IVP density per-patch population (`c20de85`)

The initial state density `Binom(S₀; N₀, s0)` used `N₀ = total_pop` (sum of all
compartments across all patches). For a 5-patch model with 440K total pop,
`Binom(5950; 440000, 0.06)` ≈ `-inf` because the expected value (26,400) is far
from the actual (5,950). The correct denominator is the per-patch population:
`Binom(5950; 100000, 0.06)`, which gives expected value 6,000 — reasonable.

Fixed by detecting stratified compartments via name suffix matching: `S_p1`
shares the `_p1` suffix with `E_p1`, `I_p1`, `R_p1`, so
`N₀ = S_p1 + E_p1 + I_p1 + R_p1 = 100,000`. Same fix applied to the stochastic
initial state draws in CSMC-AS, which were drawing
`Binom(440000, 0.06) ≈ 26,400` susceptibles per patch instead of the correct
per-patch count.

### Fix 6: Multi-stream data loader column mismatch (`8a0e9f9`)

**This was the final blocker.** `load_data_tsv` always read column 1 (the first
value column after time) regardless of stream name. With 5 observation streams
(`cases_p1` through `cases_p5`) all pointing at the same TSV file, all 5 streams
received `cases_p1`'s data.

Patches 2–5 compared their projected recovery incidence (often near zero early
in an epidemic) against patch 1's observed cases (17+ per week) → NegBinomial
`-inf`. This bug was invisible in single-stream models (only one value column)
and in forward simulation (no observation density evaluation).

Fixed by adding `load_data_tsv_column` that matches the stream name to TSV
column headers. `cases_p2` now reads the `cases_p2` column.

## Remaining issues

### Gamma density disabled

The gamma multiplier density (`log Gamma(g; dt/σ², σ²/dt)`) is disabled with
`if false {}` in `complete_data_loglik`. The gamma index tracking between
`step_one` and the density evaluator is fragile: `step_one` only pushes to
`gamma_used` when an overdispersed transition has positive rate, but the density
evaluator's index advancement logic didn't perfectly mirror this.

Disabling the gamma density is statistically valid for now — the transition
density already constrains `σ²` through `p_total = 1 - exp(-R·g·dt)`, so the
parameter is still identified. The gamma density adds a prior-like constraint
that improves mixing but is not essential. Re-enabling requires careful
alignment of gamma indexing between simulation and density evaluation.

### Merge-same-destination (architectural alternative)

An upstream colleague proposed merging transitions that share both source AND
destination compartments. In spatial models, each patch's infection transition
has the same stoichiometry (`S_i → I_i`) but different rate expressions (local
vs. importation). The Euler-multinomial split assigns flows to each sub-rate,
but only the _total_ flow matters for state evolution.

Merging these would eliminate the multinomial split density entirely for
importation groups, removing the source of floating-point fragility. This is the
"right" architectural fix but requires changes to the compiler's transition
grouping and to `step_one`'s source-group logic. Deferred to a future iteration.

## Lessons learned

1. **Clamping creates a simulation/density gap.** Any post-hoc state
   modification (clamping, balancing, events) that isn't reflected in the
   density creates a mismatch. The density must evaluate against the state that
   _generated_ the draws, not the state that _resulted_ from them.

2. **Round-trip tests are necessary but not sufficient.** The standalone density
   test (`simulate_reference` → `complete_data_loglik`) passed 100/100 seeds
   because it used the same trajectory object. The bug only manifested when the
   trajectory was reconstructed from CSMC history, which stored post-clamp
   counts. Testing density round-trips after CSMC reconstruction would have
   caught this earlier.

3. **Spatial models are qualitatively different.** Single-patch models never
   trigger negative clamping, so single-patch tests provide no coverage for this
   class of bugs. Spatial density tests should be part of the standard
   regression suite.

4. **Store the generating state, not the resulting state.** When a record needs
   to support both "what happened next" (simulation) and "how likely was what
   happened" (density), store the input state explicitly rather than deriving it
   from adjacent records. The derivation breaks when there are non-invertible
   transformations (like clamping) between steps.

5. **Golden tests must exercise the full inference pipeline.** The existing
   density round-trip tests only checked transition density (empty
   `obs_streams`, empty `observations`). They never evaluated the observation
   density or the IVP density, so fixes 5 and 6 were invisible to the test
   suite. A golden test that runs `complete_data_loglik` with actual observation
   data would have caught the column mismatch immediately.

6. **Layered bugs mask each other.** Six bugs produced the same symptom
   (`-inf`). Fixing one revealed the next. The investigation would have been
   faster with component-level diagnostics from the start — separate checks for
   IVP density, transition density, and observation density, each printing when
   it returns `-inf`. The final diagnostic framework (added during this
   incident) should be kept permanently.

7. **Multi-stream data loading is a distinct concern from multi-stream
   observation models.** The `ObsStreamSpec` + `joint_obs_weight` architecture
   was correct, but the data loader fed identical data to all streams. Testing
   multi-stream requires verifying that each stream receives its own data, not
   just that the joint likelihood computation works.

## Applicability beyond spatial models

These bugs were surfaced by a 5-patch spatial SEIR model, but patches are just
one kind of stratification. The fixes have varying applicability to other
stratified models (age, risk group, species, etc.).

### Fixes that affect ANY stratified model

**Fix 4 (deterministic check ordering)** and **Fix 6 (data column mismatch)**
are structural bugs independent of model type. Any multi-stream model — whether
streams represent patches, age groups, or surveillance sites — would hit the
data loader bug.

**Fix 5 (IVP per-patch population)** applies to any model where an IVP parameter
(`s0`) controls a compartment in one stratum. An age-stratified model with
`init { S[age1] = s0 * N_age1 }` would produce the same
`Binom(S[age1]; total_pop_all_ages, s0)` ≈ -inf. The suffix-matching fix works
for any dimension name, not just patches.

### Fixes specific to models where clamping fires

**Fixes 1–2 (counts_before snapshot, CSMC reference mismatch)** only matter when
negative clamping occurs, which requires total outflows from a compartment to
exceed its population in one substep. This needs either:

- **Multiple independent rate components** drawing from the same compartment
  (spatial importation is the canonical case), or
- **Very high rates** where $p_{\text{total}} \to 1$
  ($R_0 \cdot \gamma \cdot \Delta t$ large).

For **age-stratified models**, the typical pattern is one outflow per
age-compartment (aging from `age_i` to `age_{i+1}`), which doesn't compete with
other transitions. Clamping is unlikely unless the aging rate is very high
relative to dt.

The exception: models with both aging AND disease transitions sharing a source.
`S[age1]` with both `infection[age1]: S[age1] → E[age1]` and
`aging[age1]: S[age1] → S[age2]` are in the same source group. If both rates are
high, overdraft can occur. The fix is the same: smaller dt.

For **demographic models**, birth/death transitions are typically Poisson
(ungrouped, not in the multinomial). They don't compete with other transitions
for the same source compartment, so clamping isn't a risk.

### Fix 3 (iota / zero-rate) — depends on mixing structure

The zero-compartment problem (`I[j] = 0` makes rate exactly zero) is specific to
models where force of infection depends on a stratum-specific infectious count
that can reach zero. This includes:

- **Spatial coupling:** importation rate ∝ `I[patch_j] / N[patch_j]` → zero when
  `I[patch_j] = 0`
- **Age-structured mixing via WAIFW matrices:** FOI for age group `i` ∝
  `Σ_j C[i,j] * I[age_j] / N[age_j]` → zero when ALL age groups in the sum have
  `I = 0`
- **Multi-species models:** cross-species transmission with species- specific
  reservoirs

Models where the infection rate depends on a TOTAL infectious count (summed
across strata) don't need iota — `I_total = 0` means no infection anywhere,
which is a legitimate absorbing state.

### Recommended golden test coverage

The `seir_spatial_5_inference` golden model exercises the spatial coupling
corner case. To catch analogous bugs in other stratification dimensions:

- **Age-stratified WAIFW model** — exercises cross-age force of infection,
  age-specific IVPs, and multi-stream age-specific surveillance data. Would
  catch age-specific variants of fixes 3, 5, and 6.
- **Two-species model** — exercises cross-species transmission, which has the
  same zero-compartment structure as spatial importation.

## Hardening recommendations

These follow-up changes would prevent recurrence and catch related issues
earlier.

### 1. Euler approximation warning

Add a runtime check for $p_{\text{total}} > 0.5$ during the particle filter
validation step. When the exit probability is high, the Euler-multinomial
approximation is breaking down and the user should reduce dt.

```rust
// In step_one or the pfilter validation pass:
if p_total > 0.5 {
    n_high_p += 1;
}
// After simulation:
if n_high_p > 0 {
    log::warn!("{} substeps had exit probability > 0.5. \
        Consider reducing dt (e.g., dt = 0.25) for numerical stability.", n_high_p);
}
```

### 2. Centralize the rate epsilon — DONE (`19ac52c`)

`RATE_EPSILON = 1e-15` defined once in `chain_binomial.rs`, imported by
`pgas.rs`.

### 3. Debug assertions in step_one — DONE (`19ac52c`, `44b28d7`)

`debug_assert!(n_exit <= n_src)` in `step_one`, `simulate_reference`, and
`csmc_as` traceback.

### 4. Trace-gated -inf logging in distribution functions — TODO

Add `log::trace!` wrappers on `binom_logpmf`, `poisson_logpmf`, etc. that log
when returning `-inf`. Use `--verbosity trace` to see per-term density values
that would have identified the `k > n` binomial immediately.

### 5. Clean up existing debug diagnostics — DONE (`9e3b849`)

All density diagnostics use `log::debug!` / `log::warn!` via `--verbosity` flag.
`CAMDL_VERIFY_DENSITY` removed. Gamma density disabled with TODO.

### 6. Near-zero rate soft handling + iota warning — DONE (`faffe8f`)

Near-zero rates with nonzero flow are included in the multinomial rather than
hard-rejected. Truly zero rates emit `log::warn!` about adding iota. User-facing
guidance added to `docs/inference.md`.

### 7. Gamma density re-enablement — TODO

The gamma multiplier density `log Gamma(g; dt/σ², σ²/dt)` is disabled in
`complete_data_loglik`. Re-enabling requires aligning gamma index tracking
between `step_one` and the density evaluator. The transition density already
constrains σ² through `p_total`, so this is correctness-enhancing but not
blocking.

### 8. Euler approximation `p_total` warning — TODO

Warn when `p_total > 0.5` at any substep, signaling the Euler-multinomial
approximation is breaking down. Use `log::warn!` with count of affected
substeps.

### 9. Golden inference test — TODO

Add a golden test that runs `complete_data_loglik` with actual observation data
(not just transition density). This would have caught fix 6 (data column
mismatch) immediately. The `seir_spatial_5_inference` golden model is in place;
needs a test that loads observation data and evaluates the full LL.
