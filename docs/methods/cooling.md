# IF2 Cooling Schedule: Formula, Semantics, and the cf50 Convention

**Scope:** exactly what camdl's `cooling_fraction` parameter means, how it maps
to the perturbation-SD schedule that runs, and why scout uses a mild value
(0.70) while refine uses an aggressive value (0.05). One-stop reference for
anyone about to write a `fit.toml`, edit the cooling-related code, or interpret
an IF2 trace.

**Authoritative code:**

- Formula implementation: `rust/crates/sim/src/inference/if2.rs:250-251`
- Scout default: `rust/crates/cli/src/fit/scout.rs:14` (`SCOUT_COOLING = 0.70`)
- Refine default: `rust/crates/cli/src/fit/refine.rs:15`
  (`REFINE_COOLING = 0.05`)
- Validate cooling target: `rust/crates/cli/src/fit/validate.rs:493`
  (`cooling_target_iters: 30`)

---

## 1. The one-sentence version

camdl's `cooling_fraction` follows pomp's
[`cooling.fraction.50`](https://kingaa.github.io/pomp/manual/mif2.html)
convention: at the **halfway point** of the stage's iterations, the perturbation
SD is `cooling_fraction × initial`; at the **end**, it is
`cooling_fraction² × initial`. The parameter is stage-length- invariant — scout
with `cooling=0.70` over 30 iters and the same over 200 iters both end at 49% of
initial SD, just taking different wall-clock times.

## 2. The formula, as compiled

From `rust/crates/sim/src/inference/if2.rs:250-251`:

```rust
let total_target_steps = config.cooling_target_iters as f64 * n_obs as f64;
let per_step_cooling = config.cooling_fraction.powf(2.0 / total_target_steps);
```

And at every perturbation site (three occurrences: t=0 perturbation at line 310,
per-observation perturbation at line 374, diagnostic at line 485):

```rust
let cooling_now = per_step_cooling.powf(global_step as f64);
```

`global_step` increments once per perturbation. Each IF2 iteration consumes
`1 + n_obs` ticks (one t=0 tick plus one per observation), so after iteration
_m_ the global_step is approximately `m × n_obs` (the `+m` contribution from t=0
ticks is ≤0.1% for realistic `n_obs`).

Working through:

```
per_step_cooling = cooling_fraction ^ (2 / (target_iters × n_obs))

SD(after iter m) / SD(initial)
    = per_step_cooling ^ (global_step)
    ≈ per_step_cooling ^ (m × n_obs)
    = cooling_fraction ^ (2m / target_iters)
```

At iter = target_iters/2: exponent = 1 → SD = `cooling_fraction¹ × initial`.

At iter = target_iters: exponent = 2 → SD = `cooling_fraction² × initial`.

**The halfway-SD property is what `cf50` (cooling-fraction-at-50%) refers to.**
The name is about the _temporal_ halfway point of the iteration budget, not a
50% SD reduction.

## 3. Where `cooling_target_iters` comes from

In every production path it is set to the stage's own `n_iterations`:

- `rust/crates/cli/src/fit/runner.rs:254`
- `rust/crates/cli/src/if2.rs:299`
- `rust/crates/cli/src/profile.rs:394`

Exception: the validate stage hard-codes `cooling_target_iters = 30` regardless
of `n_iterations` (see `rust/crates/cli/src/fit/validate.rs:493`).

Consequence: `cooling_fraction = 0.70` means the same thing whether you run 30
iters or 200 iters — it's the endpoint-reduction target, not a per-iteration
rate. A longer stage takes longer to descend through the same curve, it doesn't
cool further.

## 4. The design intent: scout hot, refine cold

### Scout (`SCOUT_COOLING = 0.70`, 30 iters default)

| Point             | SD / initial                |
| ----------------- | --------------------------- |
| iter 0 (start)    | 1.000                       |
| iter 15 (halfway) | 0.700 (= cooling_fraction)  |
| iter 30 (end)     | 0.490 (= cooling_fraction²) |

Scout's perturbation SD shrinks by only 51% across the stage. The chains stay
genuinely hot throughout — their particle clouds never collapse onto a single
point. This is deliberate:

- Multiple chains are launched from dispersed starting positions drawn across
  the parameter bounds.
- Each chain explores semi-independently under moderate-SD perturbations.
- At the end of the stage, the chain's final particle cloud occupies a
  moderate-sized neighbourhood of whatever local basin the chain gravitated
  toward.
- The **cross-chain chain-agreement Â** diagnoses whether the chains agreed on a
  single basin (converged, low Â) or scattered across multiple (not converged,
  high Â — flags multi-modality). The compound scout-convergence gate combines Â
  with a loglik-eval decibans-spread check; see `docs/camdl-inference-spec.md`
  §6.1.1.

Scout's job is not to produce a single tight MLE estimate; it is to **discover
whether a single basin exists** and where it roughly is. Mild cooling preserves
the exploration capacity that makes that discovery possible.

### Refine (`REFINE_COOLING = 0.05`, 50 iters default)

| Point             | SD / initial                 |
| ----------------- | ---------------------------- |
| iter 0 (start)    | 1.000                        |
| iter 25 (halfway) | 0.050 (= cooling_fraction)   |
| iter 50 (end)     | 0.0025 (= cooling_fraction²) |

Refine's SD collapses by a factor of ~400 over 50 iterations. Chains start from
scout's best-chain parameters and progressively quench toward that point. By the
end the particle cloud is tightly concentrated near the local MLE.

Refine's job is MLE concentration, not exploration. The aggressive cooling is
what produces a crisp point estimate the user can report.

### Validate (`cooling = 0.05`, 100 iters, `target_iters = 30`)

Validate uses refine's cooling fraction but with `target_iters =
30` hard-coded
and 100 iterations. That means `cooling^2 = 0.0025` is reached at iter 30, and
iterations 30–100 run at essentially zero effective perturbation — the particle
cloud is locked at the MLE and the remaining iterations serve as a clean-ish
evaluation pass (with lingering numerical jitter that a proper final PF should
be run to replace).

## 5. Why the direction matters

The cooling-direction convention is exactly backwards from what some textbook
descriptions of simulated annealing might suggest: "exploration = slow cooling"
is a statement about _per-iteration_ cooling rate, not about _endpoint_ SD
reduction. camdl's `cooling_fraction` is an endpoint parameter, so "scout wants
slow cooling" means "scout wants a `cooling_fraction` close to 1" — i.e. the
stage ends with most of the initial SD still intact.

Equivalently: a `cooling_fraction = 0.70` scout preserves SD at ~50% of initial
through the whole stage. A `cooling_fraction = 0.05` refine _drops_ SD to 0.25%
of initial. The "aggressive cooling" phrase refers to the _final reduction_, not
the _per-iter rate_.

This is a genuinely counter-intuitive naming artifact of pomp's cf50 convention.
The convention is useful (the parameter maps directly to "what do you want the
SD to look like halfway through") but it inverts some natural-language readings.
Read the parameter as _"fraction of the initial perturbation you want to still
be alive at the halfway point."_ Scout wants that fraction high
(exploration-friendly); refine wants it low (convergence-friendly).

## 6. Empirical confirmation — instrumented run on he2010

Run parameters mirror `camdl-book/vignettes/he2010/fit_synthetic.toml`'s IF2
cooling value (0.9) but shortened to 30 iterations and 100 particles for runtime
of investigation. Effective settings: `cooling_fraction = 0.9`,
`target_iters = 30` (= n_iterations), `n_obs
= 1096` weekly obs,
`per_step_cooling = 0.999993591230`.

```
iter   0: SD = 1.0000 × initial
iter   1: SD = 0.9930 × initial
iter   2: SD = 0.9861 × initial
iter   3: SD = 0.9791 × initial
iter   4: SD = 0.9723 × initial
iter   5: SD = 0.9655 × initial
iter   6: SD = 0.9587 × initial
iter   7: SD = 0.9520 × initial
iter   8: SD = 0.9454 × initial
iter   9: SD = 0.9387 × initial
iter  10: SD = 0.9322 × initial
iter  11: SD = 0.9256 × initial
iter  12: SD = 0.9192 × initial
iter  13: SD = 0.9127 × initial
iter  14: SD = 0.9063 × initial
iter  15: SD = 0.9000 × initial    ← halfway, matches cooling_fraction exactly
iter  16: SD = 0.8937 × initial
...
iter  29: SD = 0.8157 × initial    ← end, approaches cooling_fraction² = 0.81
```

The iter-15 reading of exactly `0.9000` is the decisive empirical test. No
cooling convention other than cf50 produces that specific value at that specific
iteration. This settles the question without further argument.

_Instrumentation was a one-line `log::info!` after the existing
`cooling_at_iter` computation at `if2.rs:485`, removed after data collection._

## 7. A worked example, in full

**Question:** if I run scout with the code default `cooling_fraction = 0.70` for
30 iterations on the London measles data (1096 weekly obs), what is the
effective perturbation SD at iteration 20?

**Step 1.** Compute `per_step_cooling`:

```
per_step_cooling = 0.70 ^ (2 / (30 × 1096))
                 = 0.70 ^ (6.083e-5)
                 = exp(6.083e-5 × ln(0.70))
                 = exp(6.083e-5 × -0.3567)
                 = exp(-2.170e-5)
                 ≈ 0.9999783
```

**Step 2.** Compute `global_step` at iter 20 (ignoring the t=0 ticks, which
contribute <0.1%):

```
global_step ≈ 20 × 1096 = 21,920
```

**Step 3.** Compute effective SD:

```
SD / initial = per_step_cooling ^ global_step
             = 0.9999783 ^ 21,920
             = exp(21,920 × ln(0.9999783))
             = exp(21,920 × -2.170e-5)
             = exp(-0.4757)
             ≈ 0.6214
```

So at iter 20 of 30, the perturbation SD is about 62% of initial. Applying the
cf50 formula directly: `0.70 ^ (2 × 20 / 30) = 0.70 ^
1.333 = 0.6214`. Same
number.

## 8. Common misreadings and how to avoid them

### "The scout cooling factor is applied once per iteration"

Wrong. The `cooling_fraction` parameter is applied at the level of the endpoint
SD, not per iteration. `0.9^30` would be `0.042`, but scout with `cooling=0.9`
over 30 iters produces a final SD of 0.81, not 0.04. The `per_step_cooling`
factor is the per-iteration-step rate, and it is computed such that the endpoint
matches the cf50 convention. Don't exponentiate `cooling_fraction` by
`n_iterations`.

### "A scout run with `cooling=0.9` at 200 iterations ends with SD ≈ 10⁻¹⁰"

Wrong, and worth calling out separately because this particular miscalculation
led to incorrect design rationale in earlier proposals. The correct value is SD
≈ `0.9² = 0.81 × initial`, regardless of whether the stage is 30 iters or 200
iters.

### "Scout uses aggressive cooling because it's the exploration phase"

Wrong, but subtly. The textbook "explore slowly" rule of thumb is correct —
scout should indeed explore gently — but under the cf50 convention, "gentle"
exploration means `cooling_fraction` close to 1 (high fraction of SD preserved
at the halfway point). Saying "aggressive cooling" to mean "concentrate quickly"
is a valid natural-language statement, but it maps to `cooling_fraction`
**small**, not large. Scout wants `cooling_fraction` _high_ (~0.70); refine
wants it _low_ (~0.05).

### "I can use `cooling_fraction = 0.95` for a refine stage and get tight convergence"

No. A refine stage at `cooling=0.95` ends with SD = 0.9025 × initial, which is
barely any cooling at all. The chains will not concentrate onto a point; they'll
produce a particle cloud nearly as spread as the initial dispersal. Use
`cooling_fraction ≤ 0.10` for refine if you want meaningful MLE concentration.

## 9. Related references

- Ionides, Nguyen, Atchadé, Stoev, King (2015), _Inference for dynamic and
  latent variable models via iterated, perturbed Bayes maps_, PNAS 112(3). The
  IF2 algorithm.
- King, Nguyen, Ionides (2016), _Statistical Inference for Partially Observed
  Markov Processes via the R Package pomp_, Journal of Statistical Software,
  vol. 69. The `cooling.fraction.50` parameter semantics documented in a
  peer-reviewed form.
- pomp's `mif2()` manual page: <https://kingaa.github.io/pomp/manual/mif2.html>
  — the authoritative reference for the cf50 convention camdl inherits.

## 10. If you touch cooling code

This section exists because the cooling semantics have previously been
documented inconsistently across four surfaces and two vignette configs. If you
edit any of the following, please re-verify with an instrumented run (see §6)
and update this file accordingly:

- `rust/crates/sim/src/inference/if2.rs:250-251` (the formula itself)
- `rust/crates/cli/src/fit/{scout,refine}.rs` (the historical stage defaults)
- `rust/crates/cli/src/fit/runner.rs:254` (target_iters initialization)
- `rust/crates/cli/src/fit/validate.rs:493` (validate's fixed target_iters)

The one-line instrumented log at `if2.rs:485` used in §6 is trivial to re-apply
and gives conclusive evidence in <1 minute of runtime.
