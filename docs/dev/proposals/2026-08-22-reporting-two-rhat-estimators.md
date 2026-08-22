# Reporting two R̂ estimators, and what their disagreement means

Date: 2026-08-22\
Status: proposed\
Area: inference diagnostics (`sim/inference/convergence.rs`, `cli/fit`)\
Background assumed: none. This is the first thing to read on the topic.

## What happened

camdl fitted a two-chain PGAS model and reported:

```
beta   Rhat=6744475884463756.000 ✗   ESS bulk=3
```

The draws behind that number:

```
chain 0: 30 draws, 1 distinct value, range 0.0000e+00
chain 1: 30 draws, 1 distinct value, range 0.0000e+00
```

Both chains were exact point masses. The sampler never accepted a move. The
within-chain variance `W` is mathematically zero, so R̂ = √((n−1)/n + B/(nW))
divides by zero, and 6744475884463756 is what the floating-point arithmetic left
behind rather than a measurement of anything.

That number is not merely ugly. It sailed through every `is_finite()` check in
the pipeline, so it reached the summary as a statistic — and, through a separate
defect, the parameter it belonged to then vanished from the fit's reported
parameter list entirely. Chasing it produced this note.

## The question this note answers

camdl now computes two R̂ estimators and has to decide what to do with both:

- **classic** — Gelman & Rubin (1992), computed on the raw scale from chain
  means and variances;
- **rank-normalized** — Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021),
  computed on split half-chains after replacing each draw by the inverse-normal
  of its rank, and taken as the larger of that and its _folded_ counterpart (the
  same statistic applied to `|x − median(x)|`).

The 2021 statistic is strictly better as a _convergence test_, which is why it
is now camdl's headline. The question is whether the classic one is worth
keeping alongside it, or is legacy weight. This note answers: **keep it, report
both, and treat their disagreement as its own diagnostic** — with the evidence
for each part.

It also settles what to print when neither is defined, because the incident
above was as much a rendering failure as an arithmetic one.

## What the two estimators actually see

Three transformations separate them, and it matters that they are separate,
because each catches a different failure:

| step                             | catches                                                                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **splitting** each chain in half | a chain that drifts across its own run — invisible to a statistic that compares chain _means_                                                                 |
| **rank-normalizing**             | heavy tails and bounded parameters piling at a bound; removes the finite-variance assumption and makes the statistic invariant to monotone reparameterization |
| **folding** about the median     | chains that agree on location and disagree on _spread_                                                                                                        |

camdl's headline collapses all three into one number. That is right for a
pass/fail test and wrong for diagnosis, which is the tension this note resolves.

## What every other package does

The degenerate case above is not exotic — any stuck sampler produces it — so it
is worth knowing how the field handles it. Source read at the versions named;
every figure below was re-run locally rather than taken from documentation.

Input throughout: 2 chains × 30 draws, chain 1 ≡ `0.239349270`, chain 2 ≡
`0.438170322`.

| implementation                  | returns                     | prints in its summary table |
| ------------------------------- | --------------------------- | --------------------------- |
| Stan C++ (2.39.0, current path) | `NaN`                       | `nan`                       |
| Stan C++ (deprecated path)      | `+Inf`                      | `inf`                       |
| R `posterior` 1.7.0             | `Inf` (basic) / `NA` (rank) | `NA`                        |
| ArviZ 0.23.4 / 1.3.0            | `3372237941944279.0`        | `3.372238e+15`              |
| PyMC 5.x / 6.x (_is_ ArviZ)     | same                        | same                        |
| NumPyro 0.21.0                  | `928932217933068.2`         | overruns the column         |
| TensorFlow Probability 0.25.0   | `4.3766653e+30`             | —                           |

The field is split. Stan's current C++ guards per chain —
`chains.col(i).isApproxToConstant(...)` in `stan/analyze/mcmc/check_chains.hpp`,
whose caller returns `(NaN, NaN)`. The Python stack does not guard at all:
ArviZ's `_rhat` has only a NaN-and-shape check (`_not_valid`), and NumPyro
deliberately silences the divide-by-zero
(`with np.errstate(invalid="ignore", divide="ignore")`).

Two details are worth carrying forward.

**R `posterior`'s `NA` here is an accident, not a guard.** Its `is_constant`
test looks at the whole draws matrix, and two frozen chains at different values
are not constant. The `NA` comes from the _folded_ half: folding a symmetric
two-point set about its median gives `|a−m| = |b−m|` exactly, so the folded
draws are constant, and R's `max` propagates `NA`. With three or four frozen
chains the fold is no longer constant and `rhat()` returns `Inf`. Do not read
`posterior`'s behaviour here as a designed refusal.

**A per-chain guard was tried upstream and reverted.** `posterior` added one
(#167) and removed it four months later (#198) as "overly conservative
especially for `ess_tail`". The commented-out block survives in
`should_return_NA` with its rationale, and Vehtari's #70 states the ambiguity
plainly: constant draws could mean a parameter that is fixed, limited
floating-point accuracy, a discrete quantity with one high-probability state, or
stuck chains — and from the draws alone you cannot tell which.

## The finding that settles what to print

The large finite value is a function of the **array shape**, not the data. Five
different pairs of frozen values, ArviZ 0.23.4:

```
values (0.23934927, 0.438170322)  -> 3372237941944279.0
values (1.0, 2.0)                 -> 3372237941944279.0
values (0.25, 0.5)                -> 3372237941944279.0
values (3.0, 7.0)                 -> 3372237941944279.0
values (-11.0, 4001.0)            -> 3372237941944279.0
```

Change the shape instead and it moves:

```
(2 chains, 30 draws)   -> 3372237941944279.0
(2 chains, 100 draws)  -> 9806857692081994.0
(2 chains, 1000 draws) -> 7007069892978220.0
(4 chains, 1000 draws) -> inf
```

So an R̂ of 6.7e15 does not indicate a worse failure than one of 3.4e15. Two
unrelated broken fits produce the same digits; one broken fit at a different
draw count produces different digits. The magnitude carries no information about
the chains, which is the whole argument against displaying it as a statistic.

**camdl does not print it.** When every chain is internally constant the
estimator refuses by name. This matches Stan's current C++ and is narrower than
the `posterior` check that was reverted: that one fired when _any_ chain was
constant, camdl's only when _every_ chain is. The reported form is the reason,
not a number:

```
beta   not reported — each of the 2 chains sat at its own single value:
       the sampler never accepted a move, so R̂ has no within-chain
       variance to divide by
```

A related trap, worth stating because two implementations fell into it: ArviZ
guards constant input in `_ess` but not `_rhat`, and returns the **maximum
possible** ESS for a chain carrying no information — `ess(np.ones((4,100)))` is
`400`, pinned by a test. camdl must never do this; a frozen parameter has no
effective sample size to report.

## Why the classic estimator stays

The rank-normalized statistic is **bounded**. Ranks know only order, so however
far apart the chains are, the transformed values occupy the same finite spread.
Maximally separated chains, each internally varying by `1e-9`, `posterior`
1.7.0:

| shape       | rhat (rank) | rhat_basic |
| ----------- | ----------- | ---------- |
| m=2, n=30   | 1.8528      | 5.805e+11  |
| m=2, n=1000 | 1.8277      | 5.664e+11  |
| m=4, n=30   | 2.8452      | 1.230e+12  |
| m=4, n=1000 | 2.8405      | 1.212e+12  |
| m=8, n=30   | 4.5396      | 2.534e+12  |
| m=8, n=1000 | 4.4038      | 2.394e+12  |

The ceiling is about 1.85 for two chains, 2.84 for four, 4.5 for eight — set by
the chain count and essentially independent of the run length.

The consequence is that the headline cannot express severity. Holding the chain
separation fixed and shrinking the within-chain scale:

| within-chain sd | rhat (rank) | rhat_basic  |
| --------------- | ----------- | ----------- |
| 0               | NA          | Inf         |
| 1e-16           | 1.9004      | 1.1377e+15  |
| 1e-08           | 1.8243      | 1.21647e+07 |
| 1e-03           | 1.8148      | 115.497     |
| 1e-01           | 1.5440      | 1.46632     |

Across thirteen orders of magnitude of within-chain movement — from chains that
are frozen to floating-point resolution to chains that genuinely explore — the
rank-normalized statistic reads between 1.81 and 1.90. It cannot distinguish
"the sampler is dead" from "the sampler is mixing poorly". The classic statistic
separates those by fourteen orders of magnitude.

This is the case for keeping classic R̂: it is not a legacy fallback, it is the
only one of the two that carries scale. Both are already written to every
`*_summary.json` (`rhat`, `rhat_classic`), and `docs/workflow.md` names which is
which.

## Their disagreement is a diagnostic, and it decomposes

If two estimators of the same quantity disagree, the disagreement locates the
reason — but only if the comparison isolates one variable. camdl's two published
numbers do not: the headline is split, rank-normalized _and_ folded, the classic
one is none of those, so a single `rhat − rhat_classic` delta conflates three
independent effects and is not interpretable. Two fixture cases make that
concrete: `within_chain_drift` and `scale_disagree` both show a large gap, for
entirely different reasons, with entirely different remedies.

The fix is to publish the ladder rather than the endpoints. Four quantities,
each adding exactly one transformation:

| quantity       | chains  | scale | folded |
| -------------- | ------- | ----- | ------ |
| `rhat_classic` | unsplit | raw   | no     |
| `rhat_split`   | split   | raw   | no     |
| `rhat_bulk`    | split   | rank  | no     |
| `rhat_folded`  | split   | rank  | yes    |

`rhat` stays `max(rhat_bulk, rhat_folded)` — the reported headline is unchanged.
Three of these are already computed; `rhat_split` is computed by the fixture
generator and not yet stored by the runtime.

Each adjacent contrast then has one cause and one remedy. Measured on the
committed fixtures (`convergence_posterior_ref.tsv`, values from `posterior`
1.7.0):

| contrast                    | isolates                       | fixture that shows it                 | gap    |
| --------------------------- | ------------------------------ | ------------------------------------- | ------ |
| `rhat_split − rhat_classic` | within-chain drift             | `within_chain_drift` 1.4687 vs 1.0008 | +0.468 |
| `rhat_bulk − rhat_split`    | tail weight / non-normality    | _not demonstrated — see below_        | —      |
| `rhat_folded − rhat_bulk`   | scale vs location disagreement | `scale_disagree` 1.3130 vs 0.9984     | +0.315 |
| `rhat_classic / rhat_bulk`  | frozen chains (`W → 0`)        | the incident above                    | ~1e15  |

Read as guidance: a large **first** gap says a chain is drifting across its own
run, so lengthen warm-up or discard more of it. A large **third** gap says the
chains agree on where the posterior sits and disagree on how wide it is, which
for a particle method points at per-chain effective particle diversity. A large
**fourth** ratio says the sampler is not moving at all, which is a proposal or
step-size problem, not a "run longer" problem.

**No threshold is proposed here, deliberately.** Turning these gaps into a lint
that says "go try X" requires knowing what they look like on _healthy_ fits, and
the evidence available is twelve synthetic fixtures and one real 8-chain fit.
Picking a cutoff from that would be guessing, and camdl already has one
undecided convergence threshold in flight (gh#84: whether to adopt Vehtari et
al.'s 1.01 for the rank-normalized statistic, and an `ESS > 400` precondition).
Stacking a second undecided threshold on an undecided first is how a diagnostic
surface becomes noise. The decomposition ships now as _reported numbers_; the
lint is a named follow-up, to be designed once the gaps have been observed
across a corpus of real fits.

## What this proposes

1. **Report both estimators.** Already true; this note is the rationale, so it
   is not re-litigated later as redundancy.
2. **Store `rhat_split` alongside the existing three**, so the ladder is
   complete in `*_summary.json` and the contrasts are computable by any reader
   without re-deriving them from draws.
3. **Refuse rather than print a shape-determined number.** When every chain is
   internally constant, report the named reason. Never a large finite value, and
   never a bare blank — the blank loses the "catastrophically broken" signal,
   which was the objection that shaped this design.
4. **Make `max` propagate.** `f64::max` returns the non-NaN operand, so an
   undefined folded half is silently dropped where R's `max` propagates `NA`.
   ArviZ has the same latent inconsistency; since camdl validates against
   `posterior`, propagating is required for the oracle to mean anything.
5. **Never let a refusal delete a parameter.** A parameter with no R̂ must stay
   in the fit's reported parameter set carrying its reason. This is a type
   change — one map of a sum type per parameter, keyed by the fit's _estimated_
   set — not a patch to a derivation.

## What this does not settle

- **The reporting threshold.** gh#84, open, maintainer's call.
- **The `rhat_bulk − rhat_split` contrast is not externally demonstrated.** The
  committed fixtures show it only weakly (`heavy_tail`, gap +0.013), because the
  case was built for the headline statistic rather than for this contrast. A
  genuinely heavy-tailed fixture — Cauchy-like, where the raw-scale statistic is
  outlier-driven — is needed before that row can be relied on. Follow-up issue.
- **The divergence lint.** Deferred as argued above; follow-up issue, to be
  opened with this note cited.

## References

- Gelman, A. & Rubin, D. B. (1992). Inference from iterative simulation using
  multiple sequences. _Statistical Science_ 7(4):457-472.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021).
  Rank-normalization, folding, and localization: An improved R̂ for assessing
  convergence of MCMC. _Bayesian Analysis_ 16(2):667-718.
  [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221) — checked for a
  treatment of constant chains or `W = 0`; none found.
- `stan-dev/stan`, `src/stan/analyze/mcmc/check_chains.hpp`,
  `split_rank_normalized_rhat.hpp` (2.39.0).
- `stan-dev/posterior`, `R/convergence.R`, `R/misc.R` (1.7.0); issues #70, #167,
  #198.
- `arviz-devs/arviz`, `arviz/stats/diagnostics.py`, `arviz/stats/stats_utils.py`
  (0.23.4); `arviz-devs/arviz-stats`, `base/diagnostics.py`, `summary.py`
  (1.3.1).
- `pyro-ppl/numpyro`, `numpyro/diagnostics.py` (0.21.0); PR #412.
- `tensorflow/probability`, `python/mcmc/diagnostic.py` (0.25.0).
- Regenerate camdl's reference fixtures:
  `Rscript scripts/gen_convergence_posterior_fixture.R`.
