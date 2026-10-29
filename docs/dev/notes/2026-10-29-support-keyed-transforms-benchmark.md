# IF2's default step in z, per map family: the §12.6 benchmark

Date: 2026-10-29 Project: camdl Tags: inference, transforms, IF2, rw_sd,
benchmark

## Question

`docs/dev/proposals/2026-10-29-support-keyed-transforms.md` (D8, §12.6) fixes
IF2's perturbation step in the transformed coordinate z and gives each map
family one default step. The families are:

- `Interval`: θ = lo + (hi − lo)·σ(z);
- `LogInterval`: ln θ = ln lo + ln(hi/lo)·σ(z), for a positive kind boxed with 0
  < lo;
- the half-bounded maps: θ = lo + eᶻ or θ = hi − eᶻ.

Which step, from 2/3 down to 1/200, should each family default to? The constants
this note supports are **IF2 tuning heuristics, not optimal values**. The best
step depends on where the chains start, on the iteration budget, on the cooling
schedule and on the model. A default is a step that does acceptably across the
conditions a user is likely to meet without tuning. PMMH's and MH's default is
separate and adapted from the first warm-up step, so it is not benchmarked here.

## Setup

The harness is `tests/bench/support-transforms/`. Every run comes from one of
its Makefile targets:

| target          | what it runs                                                            |
| --------------- | ----------------------------------------------------------------------- |
| `make binaries` | pinned builds: br and ie at `877c517a`, main at `0abe59f8`              |
| `make grid`     | the base grid, 2/3 to 1/50, on every fixture, plus main                 |
| `make grid-ext` | 1/100 and 1/200 on `seir_age` and `ebola`                               |
| `make polio-r8` | `polio`, 2/3 to 1/200, seeds 1–2, 8 clean-evaluation replicates         |
| `make gh912-32` | the gh912 short-budget cell, 1/20 and 1/50 on `LogInterval`, seeds 1–32 |
| `make far`      | far starts at 30 and 150 iterations, 2/3 to 1/200                       |
| `make results`  | `results.tsv` (one row per run) and `summary.tsv` (one row per cell)    |

The three builds are:

- **br**: the branch, with `LogInterval` for a positive kind with 0 < lo;
- **ie**: the same tree with `LogInterval` disabled, so every finite box takes
  `Interval` (the D11 alternative);
- **main**: its own automatic `rw_sd`, converted to z at each particle's value.

Each candidate step c is written as an explicit natural-scale `rw_sd` that the
branch's one-time conversion turns into exactly c in z:

- `c·(hi − lo)/4` on `Interval`, converted at the midpoint;
- `c·√(lo·hi)·ln(hi/lo)/4` on `LogInterval`, converted at the geometric
  midpoint;
- `c·(θ₀ − lo)` on a lower-bounded map, converted at the start.

The families not under test take 2/3 (`Interval`), 1/5 (`LogInterval`) and 1/5
(half-bounded).

The fixtures are all chain-binomial. Models are the files named; the data are
committed under `fixtures/*/data`, and `make check-data` re-simulates them.

| fixture    | estimated (map on br)                                             | model                                            |
| ---------- | ----------------------------------------------------------------- | ------------------------------------------------ |
| `sir`      | β [0.05, 1], γ [0.01, 0.5] (`LogInterval`)                        | `tests/recovery/cases/sir`                       |
| `sir_0hi`  | γ : rate in [0, 0.5] (`Interval`)                                 | `fixtures/sir_0hi`                               |
| `sir_wide` | lb : real in [−8, 2], infection at exp(lb) (`Interval`)           | `fixtures/sir_wide`                              |
| `seir_age` | β [0.001, 0.5], σ, γ [0.01, 1] (`LogInterval`)                    | `tests/recovery/cases/seir_age`                  |
| `ebola`    | β [0.05, 2], σ [0.02, 1] (`LogInterval`), cfr [0, 1] (`Interval`) | `fixtures/ebola`                                 |
| `sir_lb`   | β, γ : rate with no range, [0, ∞) (lower-bounded)                 | `fixtures/sir_lb`                                |
| `ebola_lb` | β, σ : rate with no range (lower-bounded), cfr                    | `fixtures/ebola_lb`                              |
| `polio`    | six positive rates (`LogInterval`), ρ [0.1, 0.95] (`Interval`)    | `tests/fixtures/polio_afp_es`, its boxes         |
| `gh912`    | the same seven                                                    | the same, 2 chains × 3 iterations × 50 particles |

`ebola` is an SEIR with a case-fatality split: death occurs at rate cfr·γ and
recovery at (1 − cfr)·γ. It is fitted to weekly onsets and weekly deaths, the
two series an Ebola situation report prints. The data come from one simulation
with N = 50 000, β = 0.3, σ = 0.1, cfr = 0.6 and γ = 0.125 held. The series ends
mid-epidemic. No Ebola or spline recovery case exists in the repository; this is
the closest fixture to one.

### Conditions

The **base** condition uses each fixture's own budget. IF2 runs 8 chains × 2000
particles × 150 iterations with cooling 0.7. Starts are drawn by `lhs` over the
box; the lower-bounded fixtures, which have no box, start every chain at one
point near the optimum. `polio` runs 4 × 2000 × 50 with cooling 0.5.

The **far** conditions start every chain at one point in a corner of the box, or
far up the half-line. They run 30 or 150 iterations (`polio` 20). These are the
conditions where a larger step should win if it ever does. The corners are:

- `sir_wide`: lb = −7.5;
- `sir_0hi`: β 0.9, γ 0.45;
- `seir_age`: β 0.4, σ 0.02, γ 0.9;
- `ebola`: β 1.8, σ 0.9, cfr 0.05;
- `sir_lb`: β 4, γ 2;
- `ebola_lb`: β 3, σ 2;
- `polio`: every parameter near its upper end, ρ 0.9.

### What each cell reports

- **Final log-likelihood** is the clean evaluation in each run's
  `chain_evaluations.tsv`: a fresh particle filter at each chain's final
  estimate, with replicates. These are 8 × 4000 particles, or 8 × 1000 for
  `polio` r8 and far20 and for `gh912` r8; the base `polio` and `gh912` cells
  used 2 × 1000. It is not IF2's running value. A cell's value is the mean over
  seeds of the best chain's evaluation.
- **Exploration** is the fraction of chains whose evaluation is within 3 nats of
  the best any run on that fixture's data reached. It is shown in parentheses as
  a percentage.
- **Variability across seeds** is `sd_best_over_seeds` in `summary.tsv`.
- **Cost** is wall time per run (`mean_wall_s`), with CPU time recorded for the
  extension runs.
- **Serves** (bold) means within twice the particle-filter standard deviation of
  the best candidate in the cell's row.
- **fail k/n** counts seeds where every chain ended with no finite likelihood,
  almost always because the particle-filter watchdog stopped every chain for
  sustained loss of effective sample size.

## Measurements

`LogInterval`:

| fixture    | condition | 2/3      | 1/3         | 1/5         | 1/10         | 1/20                           | 1/50                       | 1/100            | 1/200       |
| ---------- | --------- | -------- | ----------- | ----------- | ------------ | ------------------------------ | -------------------------- | ---------------- | ----------- |
| `sir`      | base      | −60.2    | **−58.4**   | **−58.3**   | **−58.3**    | **−58.3**                      | **−58.3**                  |                  |             |
| `seir_age` | base      | −274.6   | −171.8      | −155.1      | −146.8       | −145.4                         | **−145.2**                 | **−145.2**       | **−145.2**  |
| `ebola`    | base      | −271.6   | −231.7      | −222.0      | −219.2       | −218.5                         | **−218.3**                 | **−218.3**       | **−218.3**  |
| `polio`    | r8        | −716.9   | −658.2      | −623.8      | −554.4 (14%) | −552.7 (71%)                   | **−551.8** (86%)           | **−551.2** (71%) | −567.2 (0%) |
| `gh912`    | r8        |          |             |             |              | **−690.5** (sd 116, fail 2/32) | −796.9 (sd 206, fail 1/32) |                  |             |
| `seir_age` | far150    | −280.9   | −212.8 (0%) | −154.9 (0%) | −146.9 (46%) | **−145.5** (100%)              | **−145.7** (21%)           | −150.9 (0%)      | −192.1      |
| `seir_age` | far30     | fail 3/3 | fail 3/3    | fail 3/3    | −159.7       | **−148.8**                     | −151.0                     | −189.4           | −231.6      |
| `ebola`    | far150    | −266.0   | **−233.5**  | fail 3/3    | fail 3/3     | fail 3/3                       | fail 3/3                   | fail 3/3         | fail 3/3    |
| `ebola`    | far30     | −351.7   | **−285.2**  | fail 3/3    | fail 3/3     | fail 3/3                       | fail 3/3                   | fail 3/3         | fail 3/3    |
| `polio`    | far20     | −694.2   | −702.9      | **−668.5**  | fail 1/1     | fail 1/1                       | fail 1/1                   | fail 1/1         | fail 1/1    |

`Interval` (the family's parameter only):

| fixture    | condition | 2/3             | 1/3                       | 1/5        | 1/10      | 1/20              | 1/50       | 1/100      | 1/200       |
| ---------- | --------- | --------------- | ------------------------- | ---------- | --------- | ----------------- | ---------- | ---------- | ----------- |
| `sir_0hi`  | base      | −58.7           | −58.4                     | **−58.3**  | **−58.3** | **−58.3**         | **−58.3**  |            |             |
| `sir_wide` | base      | −86.2           | −60.5                     | −62.0      | −59.2     | **−58.3**         | **−58.3**  |            |             |
| `ebola`    | base      | **−222.0**      | **−222.1**                | **−222.2** | −222.4    | **−222.2**        | **−222.2** |            |             |
| `polio`    | base      | −610.8          | −606.8                    | −625.9     | −619.1    | **−581.1**        | −588.4     |            |             |
| `gh912`    | base      | −1077           | −840                      | −930       | −1111     | **−805** (sd 102) | −829       |            |             |
| `sir_0hi`  | far150    | −58.6           | −58.4                     | **−58.3**  | **−58.3** | **−58.3**         | **−58.3**  | −58.6      | −61.1 (12%) |
| `sir_0hi`  | far30     | −84.8 (0%)      | −59.0 (17%)               | **−58.4**  | **−58.3** | **−58.3**         | −61.2 (4%) | −66.6 (0%) | −67.1 (0%)  |
| `sir_wide` | far150    | −102.6 (0%)     | **−60.5** (33%, fail 2/3) | fail 3/3   | fail 3/3  | fail 3/3          | fail 3/3   | fail 3/3   | fail 3/3    |
| `sir_wide` | far30     | **−105.7** (0%) | −163.7 (fail 2/3)         | fail 3/3   | fail 3/3  | fail 3/3          | fail 3/3   | fail 3/3   | fail 3/3    |

The `sir_0hi` far rows reached the basin in 100% of chains wherever no other
figure is shown. The `ebola` far rows for `Interval` fail at every step because
β and σ, held at the `LogInterval` provisional 1/5, die from that corner. The
`polio` and `gh912` `Interval` rows hold `LogInterval` at 1/5, which leaves them
tens of nats short whatever ρ's step is, so they say little.

Half-bounded:

| fixture    | condition | 2/3             | 1/3         | 1/5             | 1/10      | 1/20      | 1/50       | 1/100    | 1/200    |
| ---------- | --------- | --------------- | ----------- | --------------- | --------- | --------- | ---------- | -------- | -------- |
| `sir_lb`   | base      | −63.9           | −59.5       | −58.6           | **−58.3** | **−58.3** | **−58.3**  |          |          |
| `ebola_lb` | base      | −383.9          | −238.4      | −223.8          | −219.5    | −218.6    | **−218.3** |          |          |
| `sir_lb`   | far150    | −64.0 (0%)      | −59.5 (83%) | **−59.1** (92%) | fail 3/3  | fail 3/3  | fail 3/3   | fail 3/3 | fail 3/3 |
| `sir_lb`   | far30     | **−60.4** (29%) | −62.1 (0%)  | **−60.6** (42%) | fail 3/3  | fail 3/3  | fail 3/3   | fail 3/3 | fail 3/3 |
| `ebola_lb` | far150    | −394.9          | −237.9      | **−223.6** (0%) | fail 3/3  | fail 3/3  | fail 3/3   | fail 3/3 | fail 3/3 |
| `ebola_lb` | far30     | fail 3/3        | −506.9      | **−259.4** (0%) | fail 3/3  | fail 3/3  | fail 3/3   | fail 3/3 | fail 3/3 |

D11, positive kinds on `Interval` (the ie build), base condition:

| fixture    | 2/3    | 1/3    | 1/5       | 1/10      | 1/20       | 1/50       |
| ---------- | ------ | ------ | --------- | --------- | ---------- | ---------- |
| `sir`      | −60.5  | −58.4  | **−58.3** | **−58.3** | **−58.3**  | **−58.3**  |
| `seir_age` | fail   | −162.4 | −149.8    | −146.0    | **−145.3** | **−145.2** |
| `ebola`    | −256.2 | −226.2 | −220.4    | −218.4    | **−218.1** | **−218.0** |

Main reference (its automatic step), base condition:

- `sir`: −58.3;
- `sir_wide`: −147.7;
- `seir_age`: −148.5;
- `ebola`: −224.1;
- `polio`: −585.5, at seed 1 in an earlier run of the same configuration;
- `sir_0hi`: main refuses an automatic `rw_sd` on a box starting at 0.

**Cost.** Within a fixture and condition, wall time per completed run varies by
less than ±25% across steps. For example, `seir_age` far150 runs from 22 s to 32
s, and `polio` r8 from 34 s to 47 s, with the smaller steps slightly cheaper. A
failed run stops early (under 1 s). The step changes what a fit finds, not what
it costs.

## Observations

From starts spread over the box at the full budget, smaller steps give higher
final likelihoods down to about 1/50. Below that the curve is flat: `seir_age`
and `ebola` show the same value from 1/50 to 1/200. Where it moves, it turns
down between 1/100 and 1/200: `polio` gains 0.5 nats from 1/50 to 1/100 and
loses 16 nats at 1/200. Exploration agrees: `polio`'s chains in the basin rise
from 14% at 1/10 to 86% at 1/50, then fall to 0% at 1/200.

The short-budget `gh912` cell turns higher. Over 32 seeds, 1/20 ends 106 nats
above 1/50 (standard error of the difference about 42). 1/20 fails 2 of 32 seeds
and 1/50 fails 1, against 1 of 8 at main.

From a far start the result reverses in two ways.

First, small steps die. From an `ebola`, `polio`, `sir_wide`, `sir_lb` or
`ebola_lb` corner, every step at or below 1/5 (`ebola`, `sir_wide`) or 1/10
(`polio`, the half-bounded fixtures) ended with every chain stopped by the
particle-filter watchdog. The filter cannot score the corner. A swarm that moves
too little stays where every particle's weight collapses, and the watchdog ends
the chain before cooling matters.

Second, where small steps survive, the peak moves up. `seir_age` from its corner
peaks at 1/20 at both 30 and 150 iterations; 1/100 and smaller never reach the
basin. `sir_0hi` at 30 iterations peaks at 1/10–1/20; at 150 iterations it is
flat from 1/5 to 1/50.

The larger steps that survive far starts are far from the optimum at these
budgets:

- `ebola` reaches −233.5 at 1/3 against −218.3;
- `polio` reaches −668.5 at 1/5 against −551.2;
- `sir_wide` reaches −60.5 at 1/3 in one seed of three.

So no step serves every condition of any family. The far rows' best values come
from steps that fail the spread-start rows by 5 to 165 nats, and the spread
rows' best steps fail the far rows outright.

D11: on the spread-start fixtures, `Interval` and `LogInterval` on positive
kinds reach the same likelihood at the steps that serve. The geometry choice
does not move the result here.

## Interpretation

The default step trades two things. A larger step lets a swarm escape a start
the filter cannot score. A smaller step gives a precise final estimate once the
swarm is in the basin.

camdl's default starts (`uniform_unconstrained`, or `lhs`) spread chains over
the box. That is the base condition, and it favours 1/50–1/100 on `LogInterval`
and 1/20–1/50 on `Interval`. The short-budget `gh912` cell and the far starts
pull toward 1/20 or larger.

A defensible set of heuristics, each a compromise rather than an optimum:

- `Interval` 1/20: best or within noise on every spread-start fixture, including
  the wide log-β box, and the peak at 30 iterations from the `sir_0hi` corner.
- `LogInterval` 1/20 or 1/50. 1/20 is 0.2 nats short of the best on `seir_age`
  and `ebola` and 1.4 nats short on `polio`. It is better at the short budget
  (`gh912`, 106 ± 42 nats) and from the `seir_age` corner. 1/50 is the reverse.
- Half-bounded: there is no box to spread starts in, so these fixtures start
  from one point, and a far start is the realistic case. 1/5 is the only step
  that survives every far start tried. 1/10 to 1/50 are 0.3–5 nats better from a
  start near the optimum, but kill every chain from a far one.

A step large enough to escape a far start and small enough to finish precisely
is what IF2's cooling is for. The default cooling, 0.7 by iteration 50, shrinks
the step only threefold over these runs. This is a hypothesis, not tested here:
a larger starting step with faster cooling might serve both rows. That would
change the cooling default, not the step.

## Next

- The constants are the maintainer's decision. The options are set out above,
  and none is committed.
- Test the cooling hypothesis on the far rows: 1/5 cooled to 1/50 against a
  fixed 1/20.
- The far-start failures are the particle-filter watchdog stopping chains at an
  unscoreable start. A refused start under `single` is not redrawn, so a fit
  from such a start fails whatever the step.
