# IF2 cooling schedule

What camdl's IF2 `cooling` and `cooling_target_iters` settings mean, the
perturbation-SD schedule they produce, and how that schedule relates to pomp's
`cooling.fraction.50`.

**Authoritative code:** `per_step_cooling_factor` and
`cooling_multiplier_at_iter` in `rust/crates/sim/src/inference/if2.rs`; the
defaults in `rust/crates/cli/src/fit/config_v2.rs`
(`default_cooling_target_iters() = 50`). The schedule is pinned by
`rust/crates/sim/tests/if2_cooling.rs`.

## The schedule

IF2 (iterated filtering; Ionides et al. 2015) perturbs each particle's
parameters by a random walk whose standard deviation shrinks — "cools" — as the
run proceeds. The shrinkage is geometric in the number of filtering steps. Each
iteration consumes `1 + n_obs` steps: one perturbation at t = 0 and one at each
of the `n_obs` observation times. The per-step factor is

```
c = cooling ^ (1 / (cooling_target_iters × (1 + n_obs)))
```

and the perturbation SD at global step `s` is `rw_sd × c^s`. Because the step
count per iteration cancels, the multiplier at the end of iteration `m` is

```
SD(m) / rw_sd = cooling ^ (m / cooling_target_iters)
```

So the SD reaches `cooling × rw_sd` exactly at iteration `cooling_target_iters`,
and keeps shrinking at the same geometric rate after that. It does not stop at
the target and it does not depend on the stage's total `iterations`.

## Relation to pomp

This is pomp's geometric `cooling.fraction.50` schedule with
`cooling_target_iters` in place of pomp's fixed 50: the "50" in the pomp name is
a number of iterations, and `cooling.fraction.50` is the fraction of `rw.sd`
remaining after 50 of them (King, Nguyen & Ionides 2016, _J. Stat. Softw._
69(12); pomp `mif2()` manual). pomp's `mif2_cooling` returns
`alpha = cooling.fraction.50^(m/50)` at the end of iteration `m`, and
`mif2_pfilter` perturbs with `alpha × rw.sd` — exponent 1. (pomp also returns
`gamma = alpha²`, which it does not use for the perturbation.) With the default
`cooling_target_iters = 50`, camdl's schedule is pomp's.

## Worked numbers

`cooling = 0.7`, default target 50:

| end of iteration | SD / rw_sd       |
| ---------------- | ---------------- |
| 25               | 0.7^0.5 = 0.837  |
| 50               | 0.7 (the target) |
| 100              | 0.7^2 = 0.49     |
| 200              | 0.7^4 = 0.24     |

`cooling = 0.05`, target 50: 0.224 at iteration 25, 0.05 at 50, 0.0025 at 100.

Two consequences follow directly from the formula:

- Lengthening a run cools it further. A 200-iteration stage at `cooling = 0.7`
  ends at 24% of `rw_sd`, not 70%.
- Setting `cooling_target_iters` below `iterations` cools fast and then runs the
  remaining iterations near the noise floor; setting it above leaves the run hot
  at its end.

## What cooling does to the swarm

Cooling shrinks only the fresh perturbation added at each step. The parameter
swarm itself is carried whole from one iteration to the next — each particle
keeps its own θ (Ionides et al. 2015, Algorithm 1; gh#646) — so the swarm's
spread at any point is the accumulated result of every earlier perturbation and
every resampling, not the current perturbation SD alone. The per-iteration
diagnostics report both: `effective_rw_sd` (the cooled perturbation) and
`q_ratio`, its ratio to the swarm's actual spread.

## References

- Ionides, Nguyen, Atchadé, Stoev & King (2015). Inference for dynamic and
  latent variable models via iterated, perturbed Bayes maps. _PNAS_ 112(3),
  719–724. doi:10.1073/pnas.1410597112
- King, Nguyen & Ionides (2016). Statistical inference for partially observed
  Markov processes via the R package pomp. _J. Stat. Softw._ 69(12).
  doi:10.18637/jss.v069.i12
- pomp `mif2()` manual: <https://kingaa.github.io/pomp/manual/mif2.html>
