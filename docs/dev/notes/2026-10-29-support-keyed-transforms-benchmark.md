# Support-keyed transforms: the §12.6 benchmark, and why commit 6 is held

Date: 2026-10-29 Project: camdl Tags: inference, transforms, IF2, PMMH, rw_sd,
benchmark

## Question

`docs/dev/proposals/2026-10-29-support-keyed-transforms.md` §12.6 decides two
things by measurement before commit 6 (the step fixed in z) lands: D11, whether
a positive kind boxed with 0 < lo takes `LogInterval` or `Interval`, and D8's
`Interval` default step, the largest of 2/3, 1/3 and 1/5 that passes three
criteria on the `[0, hi]` fixture under its real likelihood. The proposal and
the coordinator both say: if no candidate passes, stop before commit 6 and
return the numbers to the maintainer. This note records the numbers.

## Setup

Three binaries, each with a benchmark-only patch (not committed) that writes
IF2's final particle swarm to a file per chain:

- **main**: `origin/main` at `0abe59f8`, the walled `Log` / `Logit` maps with
  the natural-scale step converted at each particle;
- **LogInterval**: the commit-6 tree (branch `wip/support-transforms-c6`), the
  proposal as written;
- **Interval everywhere**: the same tree with `LogInterval` disabled, the D11
  alternative.

The 0.5× and 2× steps and the three `Interval` candidates are given as an
explicit natural-scale `rw_sd`. Series (i) converts an explicit `rw_sd` once at
the box's midpoint, so `rw_sd = c·(hi − lo)/4` is a step of exactly `c` in z on
`Interval`, and `c·√(lo·hi)·ln(hi/lo)/4` on `LogInterval`. On main the scale
multiplies main's own automatic natural step.

Fixtures, all on synthetic data from `tests/recovery/cases/*/truth.toml`
(`simulate --backend chain_binomial --seed 1`):

| fixture       | what it is                                                                                   |
| ------------- | -------------------------------------------------------------------------------------------- |
| `seir_age`    | the recovery case; three rates boxed `[0.001, 0.5]`, `[0.01, 1]`, `[0.01, 1]`                |
| `sir`         | the recovery case; `beta` on `[0.05, 1]`, `gamma` on `[0.01, 0.5]`                           |
| `sir_near`    | `sir` with `gamma` boxed `[0.01, 0.16]`                                                      |
| `sir_0hi`     | `sir` with `gamma : rate in [0.0, 0.5]`, the `[0, hi]` fixture (`Interval` in both variants) |
| `sir_flat0hi` | `sir_0hi` with `gamma` kept out of the dynamics: the flat-likelihood run, reported only      |
| `sir_wide`    | `sir` with `lb : real in [-8, 2]` and infection at `exp(lb)`: a wide box on a log quantity   |
| `ode`         | the `mh_ode_recovery.rs` model under `mh`, `beta` on `[0.05, 5]`                             |

`sir_wide` was added at the coordinator's request: on the B-spline branch a
`real` log-coefficient on `[−8, 2]` with the automatic step `(hi − lo)/6 = 1.67`
drove every IF2 particle out of the feasible region, where `rw_sd = 0.05` works;
D8's `Interval` step of 2/3 reproduces that natural step at the midpoint.

Budgets: IF2 8 chains × 2,000 particles × 150 iterations, cooling 0.7, `lhs`
starts (the recovery fixtures' own budget); PMMH 4 chains × 1,500 iterations,
500 burn-in, 300 particles, `single` start; MH 4 × 3,000, 1,000 burn-in. Seed 1
throughout; the deciding cases were repeated at seeds 2 and 3. Another session's
`make test` and this branch's `make test-fast` ran during parts of the run, so
wall-clock ESS/s carries load noise of up to ±50% (the same seed-1 PMMH case
took 6.7 s and 10.1 s); ESS per kept draw is reported beside it, and the cost
per draw does not depend on the step.

Columns: IF2 _best ll_ and _mean ll (sd)_ are the clean evaluations of each
chain's final estimate (8 replicates × 4,000 particles); _PF sd_ is the sd of
one evaluation (`se`·√8); _near lo / hi_ the largest fraction, over chains, of
the final swarm within 10⁻³ of the width of an end; _min sd(z)_ the smallest
swarm sd in the variant's own z.

## Measurements

### The `[0, hi]` fixture decides D8

`sir_0hi`, real likelihood (`gamma`'s maximum is interior, 0.25):

| constant | IF2 best ll, seeds 1 / 2 / 3 | PF sd | near ends | min sd(z) | PMMH ESS/s, quiet seeds 1 / 2 / 3 | ESS per draw, seeds 1 / 2 / 3 |
| -------- | ---------------------------- | ----- | --------- | --------- | --------------------------------- | ----------------------------- |
| 2/3      | −58.71 / −58.63 / −58.70     | 0.04  | 0 / 0     | 0.58      | 37.2 / 33.4 / 32.6                | 0.094 / 0.091 / 0.086         |
| 1/3      | −58.33 / −58.36 / −58.37     | 0.035 | 0 / 0     | 0.41      | 37.2 / 27.8 / 40.3                | 0.091 / 0.073 / 0.101         |
| 1/5      | −58.30 / −58.30 / −58.29     | 0.03  | 0 / 0     | 0.32      | 29.3 / 27.3 / 27.5                | 0.073 / 0.095 / 0.071         |
| main     | −58.31 (seed 1)              | 0.04  | 0 / 0     | 0.00      | 3.5 (seed 1)                      | 0.009                         |

main's swarm sd of 0 is the defect the interim fix refuses: the automatic step
on a walled `Log { 0, hi }` is ~10⁻¹⁴⁹, so `gamma` never moves.

Applying the rule (largest constant meeting all three):

- **2/3 fails** the IF2 criterion at every seed: 0.33–0.41 below the best of the
  three, against 2·PF sd ≈ 0.08.
- **1/3** is within 2·PF sd at seeds 1 and 2 (0.03, 0.06) and outside it at seed
  3 (0.08 > 0.064). Its PMMH ESS/s is 0.77 of 2/3's in the loaded seed-1 run and
  1.0 in the quiet repeat; on the three quiet seeds its mean is the highest
  (35.1 against 34.4 and 28.0).
- **1/5** is the best IF2 result; its quiet ESS/s is 0.80 of the best mean (28.0
  / 35.1), on the line.

So at the specified seed, with ESS/s from the quiet repeat, **1/3 is the
constant the rule selects**; it misses the IF2 criterion at one of three seeds
by 0.016 nats, and with the loaded seed-1 timing it misses the ESS/s criterion.
The selection is not robust.

The flat-likelihood run (reported, not a criterion) puts 95% / 81% of the swarm
within 10⁻³·w of the ends at 2/3, 67% / 67% at 1/3, 46% / 23% at 1/5.

### The wide `real` box fails every candidate

`sir_wide`, IF2 best clean log-likelihood at seeds 1 / 2 / 3, and PMMH
acceptance:

| step in z                   | IF2 best ll              | PF sd     | PMMH acceptance    |
| --------------------------- | ------------------------ | --------- | ------------------ |
| 2/3 (natural 1.67)          | −70.6 / −96.3 / −91.9    | 2.1–2.5   | 0.71 / 0.69 / 0.72 |
| 1/3                         | −61.2 / −60.3 / −59.9    | 0.17–0.27 | 0.59 / 0.60 / 0.61 |
| 1/5                         | −61.8 / −62.3 / −61.9    | 0.20–0.26 | 0.55 / 0.55 / 0.50 |
| 0.02 (natural 0.05)         | −58.35 / −58.31 / −58.37 | 0.03      | 0.39 / 0.38 / 0.38 |
| main, natural 1.67 (seed 1) | −174.7                   | 2.5       | 0.72               |

Against main every candidate is better, so §12.6's IF2 criterion, which is
relative to main, passes. Against the log-likelihood a small step attains, every
candidate is 1.5–38 nats short, and every candidate's PMMH acceptance is above
the 0.5 bound. The coordinator's instruction for this case is to stop before
commit 6 if every candidate fails here.

### Other criteria that do not pass for `LogInterval` as written

| criterion (§12.6)                                      | `LogInterval`          | main            |
| ------------------------------------------------------ | ---------------------- | --------------- |
| `seir_age` IF2 best ll, best of three steps            | −146.30 (×0.5)         | −145.77 (×0.5)  |
| `sir` IF2 best ll, best of three steps                 | −58.27 (×0.5)          | −58.26 (×0.5)   |
| `sir_near` swarm within 10⁻³·w of `hi`, ×0.5 / ×1 / ×2 | 76% / 99% / 97%        | 11% / 15% / 20% |
| `sir_near` PMMH ESS per draw, ×1                       | 0.044                  | 0.069           |
| PMMH acceptance in [0.1, 0.5]                          | 0.545 at `seir_age` ×2 | all inside      |

- `seir_age`: 0.53 nats below main at the best step, against 2·PF sd of
  0.15–0.18. The fixture is bimodal and the chains spread (sd 1.2 nats at ×0.5),
  so this rests on one seed.
- `sir_near`: with these data `gamma`'s maximum on `[0.01, 0.16]` is on the
  upper bound (the unconstrained maximum is 0.255), and 6 of 8 `LogInterval`
  chains found a second basin (`beta` 0.26, `gamma` 0.025, ll −59.05) that main
  missed. The chains on the bound mode drew the swarm to z ≈ 8, within 10⁻³·w of
  `hi`: §9.1's "a steep rise draws the swarm far out". The swarm criterion (≤
  max(5%, main's)) reads this as a pile-up. Its sd in z is 0.49, so it has not
  collapsed. Whether the criterion should apply to a maximum on the bound is a
  question about the criterion.
- `ode` (MH): all variants within noise (acceptance 0.49–0.50, ESS per draw
  0.21–0.24).

### D11: `Interval` everywhere is worse

`Interval` for every finite box refused `seir_age` IF2 at ×1 and ×2 (every
chain's final estimate −∞; at ×1 the steps are 2/3 in z on three rates) and was
worse than `LogInterval` on `sir` and `sir_near` IF2 at every step; its PMMH
acceptance exceeded 0.5 in all six `sir`, `sir_near` and `seir_age` cells at ×1
and ×2. `LogInterval` is the better map for positive kinds with 0 < lo.

## Interpretation

The rule picks **1/3** for the `Interval` constant at the specified seed, but
not robustly, and **no candidate passes on a wide box over a log-scale
quantity**: the default step in z is a fraction of the width, and on `[−8, 2]` a
step of 1/5 to 2/3 in z is 0.5 to 1.67 in `lb` at the midpoint, a factor of e⁰·⁵
to e¹·⁷ in `exp(lb)` per perturbation. The fixed step in z removes the
end-of-box blow-up (§9.1) and makes a positive kind on `[0, hi]` move, but it
does not supply a scale where the box is much wider than the region the data
allow.

## Next

Commit 6 and series (ii) are held on `wip/support-transforms-c6` (local). The
maintainer's decisions are listed in the hand-off report.
