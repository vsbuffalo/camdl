# IF2's default step in z, per map family: the §12.6 benchmark

Date: 2026-10-29 Project: camdl Tags: inference, transforms, IF2, rw_sd,
benchmark

## Question

`docs/dev/proposals/2026-10-29-support-keyed-transforms.md` (D8, §12.6) fixes
IF2's perturbation step in the transformed coordinate z and gives each map
family one default: `Interval` (θ = lo + (hi − lo)·σ(z)), `LogInterval` (ln θ =
ln lo + ln(hi/lo)·σ(z), a positive kind boxed with 0 < lo) and the half-bounded
maps (θ = lo + eᶻ or hi − eᶻ). Which constant, from 2/3, 1/3, 1/5, 1/10, 1/20
and 1/50, serves every fixture of its family? A constant serves a fixture when
IF2's best final log-likelihood (mean over seeds) is within twice the
particle-filter standard deviation of the best candidate; on the gh912
short-budget cell it must also leave no more seeds with every chain at −∞ than
main does. PMMH's and MH's default is separate and adapted, so it is not
benchmarked.

## Setup

Binaries (benchmark only, nothing patched into the branch):

- **br**: branch `feat/support-transforms` at `877c517a` (step fixed in z,
  `LogInterval` for a positive kind with 0 < lo);
- **ie**: the same tree with the `LogInterval` arm of `Transform::for_domain`
  disabled, so every finite box takes `Interval` (the D11 alternative);
- **main**: `origin/main` at `0abe59f8`, its own automatic `rw_sd` converted at
  each particle's value (the reference).

Every step is written as an explicit natural-scale `rw_sd` chosen so the
branch's one-time conversion gives exactly the candidate in z: `c·(hi − lo)/4`
on `Interval` (converted at the midpoint), `c·√(lo·hi)·ln(hi/lo)/4` on
`LogInterval` (the geometric midpoint), `c·(θ₀ − lo)` on a lower-bounded map
(converted at the start). Families not under test take 2/3 (`Interval`), 1/5
(`LogInterval`) and 1/5 (half-bounded).

Fixtures, all chain-binomial:

| fixture    | parameters estimated (map on br)                                           | data                                                 |
| ---------- | -------------------------------------------------------------------------- | ---------------------------------------------------- |
| `sir`      | β [0.05, 1], γ [0.01, 0.5] (`LogInterval`)                                 | the recovery case's weekly cases                     |
| `sir_0hi`  | β (`LogInterval`), γ : rate in [0, 0.5] (`Interval`)                       | the same                                             |
| `sir_wide` | lb : real in [−8, 2], infection at exp(lb) (`Interval`); γ (`LogInterval`) | the same                                             |
| `seir_age` | β [0.001, 0.5], σ, γ [0.01, 1] (`LogInterval`)                             | the recovery case's two age streams                  |
| `ebola`    | β [0.05, 2], σ [0.02, 1] (`LogInterval`), cfr [0, 1] (`Interval`)          | weekly onsets and weekly deaths, 140 days, simulated |
| `sir_lb`   | β, γ : rate with no range, [0, ∞) (lower-bounded)                          | `sir`'s data                                         |
| `ebola_lb` | β, σ : rate with no range (lower-bounded), cfr [0, 1]                      | `ebola`'s data                                       |
| `polio`    | six positive rates (`LogInterval`), ρ [0.1, 0.95] (`Interval`)             | `tests/fixtures/polio_afp_es`, its `fit.toml` boxes  |
| `gh912`    | the same seven                                                             | the same, 2 chains × 3 iterations × 50 particles     |

`ebola` is an SEIR with a case-fatality split (death at cfr·γ, recovery at (1 −
cfr)·γ, γ = 0.125 held), N = 50 000, β = 0.3, σ = 0.1, cfr = 0.6, ρ = 0.7; its
data are one simulation at seed 1, ending mid-epidemic as a situation-report
series does. No Ebola or spline recovery case exists in `tests/recovery/cases`;
this is the closest fixture to one.

Budgets: IF2 8 chains × 2000 particles × 150 iterations, cooling 0.7, `lhs`
starts (`single` on the lower-bounded fixtures, which have no box to spread in);
seeds 1–3 (`seir_age` 1–2). `polio`: 4 chains × 2000 × 50, cooling 0.5, seed 1.
`gh912`: seeds 1–8. The final log-likelihood is the clean particle-filter
evaluation in `chain_evaluations.tsv`. Commands and binaries are in the run
directory's `README` and `bench2/bench.py`; the per-run table is
`bench2/results.tsv`.

## Measurements

Mean best final log-likelihood over seeds; **bold** where the candidate serves.
The particle-filter standard deviation is 0.03–0.3 on every fixture except
`polio` and `gh912` (about 1).

`LogInterval`:

| fixture         | 2/3     | 1/3        | 1/5        | 1/10       | 1/20       | 1/50        | main    |
| --------------- | ------- | ---------- | ---------- | ---------- | ---------- | ----------- | ------- |
| `sir`           | −60.21  | **−58.43** | **−58.31** | **−58.27** | **−58.26** | **−58.26**  | −58.27  |
| `seir_age`      | −274.61 | −171.76    | −155.07    | −146.76    | −145.38    | **−145.19** | −148.48 |
| `ebola`         | −271.59 | −231.73    | −221.97    | −219.18    | −218.50    | **−218.33** | −224.10 |
| `polio`         | −716.89 | −656.42    | −610.79    | −555.22    | −552.06    | **−551.74** | −585.5  |
| `gh912`, mean   | −1896   | −1775      | −1077      | −780       | **−642**   | −871        | −832    |
| `gh912`, failed | 0/8     | 0/8        | 1/8        | 1/8        | 1/8        | 1/8         | 1/8     |

`Interval`, branch maps (the family's parameter only):

| fixture         | 2/3         | 1/3         | 1/5         | 1/10       | 1/20        | 1/50        |
| --------------- | ----------- | ----------- | ----------- | ---------- | ----------- | ----------- |
| `sir_0hi` (γ)   | −58.68      | −58.35      | **−58.30**  | **−58.29** | **−58.29**  | **−58.28**  |
| `sir_wide` (lb) | −86.24      | −60.47      | −61.97      | −59.25     | **−58.34**  | **−58.34**  |
| `ebola` (cfr)   | **−221.97** | **−222.10** | **−222.19** | −222.42    | **−222.21** | **−222.18** |
| `polio` (ρ)     | −610.79     | −606.82     | −625.93     | −619.11    | **−581.09** | −588.40     |
| `gh912`, mean   | −1077       | −840        | −930        | −1111      | **−805**    | −829        |
| `gh912`, failed | 1/8         | 1/8         | 0/8         | 1/8        | 0/8         | 0/8         |

`Interval` on every box (the `ie` variant, D11):

| fixture    | 2/3     | 1/3     | 1/5        | 1/10       | 1/20        | 1/50        |
| ---------- | ------- | ------- | ---------- | ---------- | ----------- | ----------- |
| `sir`      | −60.52  | −58.36  | **−58.29** | **−58.27** | **−58.25**  | **−58.26**  |
| `seir_age` | −inf    | −162.42 | −149.75    | −146.00    | **−145.30** | **−145.19** |
| `ebola`    | −256.18 | −226.22 | −220.41    | −218.44    | **−218.09** | **−218.03** |

Half-bounded:

| fixture    | 2/3     | 1/3     | 1/5     | 1/10       | 1/20       | 1/50        |
| ---------- | ------- | ------- | ------- | ---------- | ---------- | ----------- |
| `sir_lb`   | −63.88  | −59.47  | −58.62  | **−58.28** | **−58.26** | **−58.26**  |
| `ebola_lb` | −383.88 | −238.40 | −223.85 | −219.53    | −218.60    | **−218.33** |

Main fails `sir_0hi` outright: it refuses an automatic `rw_sd` on a positive
kind boxed `[0, hi]`. Main's `sir_wide` mean is −147.68.

Iterations until a chain's log-likelihood is within one unit of its final value
fall as the step shrinks on the fixtures that are hard to fit: `seir_age` 136
(1/5) to 25 (1/50), `ebola` 140 to 65, `ebola_lb` 137 to 54. Smaller steps do
not converge more slowly at this budget.

## Observations

At the fixtures' own budgets, smaller steps give higher final log-likelihoods in
every family. The present midpoint equivalents (2/3 on `Interval`, 1/5 on
`LogInterval`) are 3.6–59 nats short on `seir_age`, `ebola` and `polio`.
`seir_age`, `ebola`, `ebola_lb` and `polio` peak at 1/50, the smallest candidate
tested, so the optimum may lie below the grid.

`Interval`: 1/20 serves every fixture, both narrow boxes (`sir_0hi`, `cfr`) and
the wide log-β box (`sir_wide`, where 2/3 is 28 nats short), and the positive
kinds under the `ie` variant. The `polio` ρ column is one seed, with
`LogInterval` held at 1/5, so it says little.

`LogInterval`: 1/50 serves `sir`, `seir_age`, `ebola` and `polio` and fails no
more `gh912` seeds than main (1 of 8). On `gh912` its mean is 229 nats below
1/20, but that cell's per-seed spread is several hundred nats (1/50 runs from
−562 to −1526), so the mean does not separate 1/20 from 1/50. 1/20 is 0.2–0.3
nats below 1/50 on `seir_age`, `ebola` and `polio`, which is more than twice the
particle-filter standard deviation there.

Half-bounded: 1/50 serves both. 1/20 is 0.27 nats short on `ebola_lb`.

D11: at the steps that serve, `Interval` and `LogInterval` on positive kinds
reach the same likelihood (`seir_age` −145.19 both; `ebola` −218.03 against
−218.33, with cfr's step differing between the two runs). The geometry choice
does not move the result here.

## Interpretation

Under the rule as written, the constants are `Interval` 1/20, `LogInterval` 1/50
and half-bounded 1/50. The `LogInterval` constant is not settled: the
long-budget fixtures prefer 1/50, and `gh912`'s mean prefers 1/20 within noise
it cannot resolve. Each of these constants is also at or near the bottom of the
grid. These results invert the premise that a larger step explores faster under
the same cooling. At 150 iterations with lhs starts, the swarm finds the basin
from the starts, and a large step mostly adds noise to the final estimate. A fit
started far from its optimum with few iterations might favour larger steps, and
none of these fixtures does that except `gh912`.

## Next

- The constant is the maintainer's call (the proposal returns it when a fixture
  disagrees): `Interval` 1/20, half-bounded 1/50, and `LogInterval` 1/20 or
  1/50.
- Extend the grid to 1/100 and 1/200 on `seir_age`, `ebola` and `polio`, to find
  where the curve turns.
- Run `gh912` at 32 seeds for 1/20 and 1/50; that is the run that would separate
  the two.
