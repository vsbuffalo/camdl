# Where a PMMH likelihood spends its time on an age-structured measles model, and what BTRS and the flat evaluator buy there

Date: 2026-11-16\
Project: camdl\
Tags: profiling, pmmh, chain-binomial, binomial, btrs, flat-eval, samply

Sibling to
[`2026-08-24-pgas-binomial-sampler-is-half-the-fit.md`](2026-08-24-pgas-binomial-sampler-is-half-the-fit.md)
(the same question on a small two-province Ebola model under PGAS) and
[`2026-06-14-flat-bytecode-evaluator.md`](2026-06-14-flat-bytecode-evaluator.md)
(`CAMDL_EVAL_FLAT`). This note measures both levers on a third model, in a
different regime: a long series with a few large compartments, fitted by PMMH.

## The workload

`camdl-london-measles/models/m4n_noise.camdl`: measles in the County of London,
1950–1964, an SEIR model in three age groups (under 5, 5–14, 15 and over) plus a
maternally protected class, with gamma white noise (`overdispersed(...)`) on
each age group's infection rate. Fitted by PMMH, 4 chains × 2,000 particles ×
2,000 iterations, dt = 0.5 day, training window 1950-01-01 to 1960-12-31 (about
8,000 substeps).

Per particle-substep the chain-binomial step makes **25 binomial draws**: one
total-exit draw for each of the 13 source compartments, plus one split draw for
each of the 12 compartments with two exits. Every compartment with a second exit
has it only because of a background death transition (μ = 3.91 × 10⁻⁵ per day);
without deaths the step would make 10 draws.

## Finding 1 — one likelihood is ~28 CPU-s, ~1.7 µs per particle-substep

A serial `camdl pfilter` at a PMMH start draw, 2,000 particles, took 38 CPU-s
over the full 1950–1964 span. The fit scores the 4,017-day training window of
that 5,480-day span, so one likelihood costs ≈ 28 CPU-s (scaled, not measured on
the truncated window) and an iteration of four chains ≈ 112 CPU-s.

The live fit ran at 22 s per iteration (158 iterations in 58 min). The machine
was running a second PMMH fit at the same time, both sized to all 16 cores (load
average 74). That contention is why the wall-clock figures in this note are not
used; every comparison below is CPU time, which contention does not inflate.

## Finding 2 — the profile: sampling ~40%, rate evaluation ~23%

samply, `pfilter --parallel 4`, 2,000 particles, CPU-weighted samples on the
four Rayon workers (49.3 CPU-s):

| share (inclusive) | what                                                              |
| ----------------- | ----------------------------------------------------------------- |
| 83.5%             | `chain_binomial::step_one`                                        |
| 26.3%             | `StatefulRng::binomial` (BINV inline, plus the BTPE call)         |
| 18.1%             | `rand_distr` BTPE                                                 |
| 8.3%              | ChaCha8 block generation                                          |
| 22.7%             | `eval_propensities` (rate evaluation, `eval_resolved` 22.2%)      |
| 15.4%             | Rayon workers spinning or sleeping at the per-observation barrier |
| 2.6%              | `__powidf2` (`q.powi(n)` in BINV)                                 |

The barrier share is inflated by the oversubscribed machine and is not a
measurement of the filter's own overhead.

## Finding 3 — the levers, measured

One likelihood, 500 particles, one thread (`--parallel 1`), seed 1, full span
with `--allow-forcing-extrapolation`. CPU seconds, two interleaved repeats each,
agreeing within 1.5%. BTRS was measured by a build with BTRS hard-wired into
`bootstrap_filter`; the shipped `--binomial btrs` reproduces its loglik exactly
(−6339.1195).

| variant                                          | CPU s | speedup | loglik    |
| ------------------------------------------------ | ----- | ------- | --------- |
| BTPE, tree-walk evaluator (the old default)      | 9.5   | 1.00×   | −6340.016 |
| `CAMDL_EVAL_FLAT=1`                              | 9.1   | 1.06×   | −6340.016 |
| BTRS                                             | 8.8   | 1.09×   | −6339.120 |
| BTRS + flat                                      | 8.2   | 1.16×   | −6339.120 |
| + the three R-compartment deaths `deterministic` | 7.3   | 1.31×   | −6341.553 |
| no deaths at all (an upper bound, not a variant) | 6.6   | 1.45×   | −6347.873 |

The flat evaluator leaves the loglik bit-identical, as it did on the Ebola model
(1.07×, `2026-08-24` note). BTRS draws different values from the same stream, so
its loglik differs by Monte Carlo noise.

## Interpretation

Neither lever is large on this model, and neither could be: even with binomial
sampling and rate evaluation both free, the step's remaining work bounds the
speedup at roughly 3×. The cost is the size of the job — 16 million
particle-substeps per likelihood, 25 draws each — and dt = 0.5 doubles it. A
larger factor needs fewer likelihood evaluations or a different structure (SIMD
across particles), not a faster scalar step.

Both levers are nonetheless free once built, so this note is the measurement
behind making BTRS selectable for PMMH, IF2, `pfilter` and `profile` (the
sampler had been reachable from PGAS only), and behind making the flat evaluator
the only rate-evaluation path.
