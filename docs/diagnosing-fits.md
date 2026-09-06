# Diagnosing a fit that won't behave

A fit can fail in two fundamentally different ways, and they call for opposite
responses. Either the **model** is misspecified — it cannot generate data that
looks like what you observed, no matter the parameters — or the **inference** is
failing — the model is fine but the sampler or filter cannot navigate to the
answer. The symptom you see (chains that won't mix, a likelihood that won't
climb, an estimate pinned at a bound) usually does not tell you which one you
have. Tuning the sampler when the model is wrong wastes hours; fixing the model
when the sampler is the problem does nothing. The first job is therefore
**diagnosis, not tuning**.

This page is the decision tree. For how the particle filter, IF2, PGAS, and NUTS
actually work — and what each diagnostic column means — see
[`camdl docs inference`](inference.md); this page assumes that mechanics and
focuses on _which question to ask first_.

## 1. First question: is it the model or the inference?

Do not touch a sampler knob until you know. The single highest-return test is
**synthetic self-consistency**: simulate data from the model at a plausible
parameter vector θ, then re-fit (or re-filter) and see whether you recover θ.

```bash
# Generate synthetic data at a known θ
camdl simulate model.camdl --params theta.toml --obs synth.tsv --seed 1

# Re-fit with a fit.toml whose [data] points at synth.tsv, then compare the
# recovered estimate to theta.toml. (Data path lives in the fit.toml's
# [data] block — there is no --data flag on `camdl fit run`.)
camdl fit run fit_synth.toml --seed 2
```

Read the result this way:

- **Recovers θ on synthetic but fails on the real data → misspecification.** The
  inference machinery works; the model cannot reproduce the real data. Stop
  tuning and fix the model (a missing mechanism — seasonal forcing, a second
  introduction, reporting structure — is the usual culprit).
- **Fails even on synthetic → it's the inference.** The data came from exactly
  this model, so any failure to recover θ is the filter/sampler, not the model.
  Now sampler diagnostics are worth your time.

This test belongs **early**, before any sampler tuning — it flips the entire
diagnosis and costs one simulate-plus-fit.

**Its blind spot: a defect shared by the generator and the scorer.** The test
compares one code path against another, so it can only see a defect that one of
them has and the other does not. When the same wrong behaviour sits on both
sides, simulate-then-refit recovers θ cleanly from data that was never right.
Concretely (gh#681): a real-valued compartment referenced in an observation
evaluates as zero on **both** the forward emitter and the value-path scorer, so
a model whose observation adds an environmental reservoir generates data with
the reservoir dropped, scores it with the reservoir dropped, and returns θ to
three digits. Green test, wrong model.

The general rule: **a self-consistency test cannot see a defect its two halves
share; a cross-path comparison can.** Two paths that are supposed to compute the
same quantity — `mh` and `nuts` on one model and one dataset — disagree when one
of them is wrong, and that disagreement is the only signal a shared defect
leaves. Keep running the self-consistency test first; reach for the cross-path
check when it passes and you still do not believe the fit.

## 2. "Looks fittable" ≠ "is fittable"

A likelihood landscape that looks smooth and peaked can still be unfittable, for
two reasons:

1. **A central value per grid point hides the per-evaluation noise the sampler
   actually eats.** For a stochastic model the likelihood at a fixed θ is itself
   a noisy Monte-Carlo estimate; a landscape that reports only a summary smooths
   that away.
2. **A likelihood-only landscape sampled over bounds ignores the prior**, so a
   direction the prior would downweight still looks freely explorable.

In camdl, `camdl survey` is exactly this likelihood landscape — it draws points
by Latin-hypercube over the declared `[estimate]` bounds. Two things to know:

- It does **not** silently smooth the noise. By default it runs several particle
  replicates per point and reports `loglik_se` (the replicate standard error on
  the log scale) and `mean_ess` alongside `loglik`, summarises the across-point
  SE distribution in `summary.json`, and warns when too many points exceed the
  ~1.7-nat reliability bar (Doucet et al.). **Read `loglik_se` and `mean_ess`,
  not just the shape of `loglik`.** A peak built from points with large SE or
  collapsed ESS is an artifact.
- It is **prior-free by construction**. The landscape is the likelihood, not the
  posterior, so a flat direction a prior would tame still looks explorable. Heed
  the bound-clustering warning (top points pinning against a bound) — that is
  the landscape telling you a direction is unidentified by the data alone.

## 3. Two inference failure modes — opposite fixes

If §1 pointed at the inference, separate these two, because their fixes are
unrelated.

**(a) An under-resourced particle filter — PMMH, IF2 _and_ PGAS.** Every
particle method degrades when the filter has too few particles. The **lever is
the same in all of them (more particles)**, but the **symptom differs by
method**, and reading the wrong symptom sends you to the wrong fix.

- **PMMH and IF2 consume the _marginal_ likelihood**, so their symptom is
  _noise_: the filter's log-likelihood estimate is too variable, acceptance
  collapses, and IF2's particle ranking becomes arbitrary.
- **PGAS conditions on a sampled latent path instead of marginalizing it out, so
  its symptom is _mixing_, not noise.** The conditional sequential Monte Carlo
  (CSMC) step has to renew the reference trajectory; with too few particles it
  rarely does, the chain barely moves in latent space, and what you see is
  **poor R̂ and a renewal profile that is flat near zero over the early part of
  the series** — never a noisy loglik. Read the **profile**, not the aggregate
  (`path_renewal` in `pgas_summary.json`, printed at the end of every PGAS
  stage). The **filter ESS** block beneath it (`filter_ess`, table
  `filter_ess.tsv`) is the direct reading: it says, per observation, how many
  particles the resample actually drew from.

**Read the profile, not `trajectory_renewal` alone.** The aggregate is a
weighted mean over ten equal time bins, and its late bins are high in most runs:
the traceback's lineages have not yet coalesced by the time it reaches the late
states, so the tail of the path renews freely and holds the mean up. A run whose
early path is completely frozen therefore still averages a healthy-looking
third. Measured on an 11-compartment stochastic Ebola model, 103 daily
observation times, post-burn-in means across 16 chains:

```text
b0    b1    b2    b3    b4    b5    b6    b7    b8    b9   trajectory_renewal
0.03  0.03  0.03  0.03  0.03  0.03  0.53  0.84  0.87  0.98        0.336
```

An aggregate of 0.34 reads as "a third of the path renews per sweep". What is
happening is that the first sixty percent of the series — the whole initial
condition and the early dynamics — changes in 3% of sweeps, because the CSMC
genealogy has coalesced and the early path is held at the reference. The
parameters whose likelihood lives in that prefix (initial infectious load,
routing shares, dwell durations) sat at R̂ 2.0-3.7 while every observation-model
parameter, whose likelihood is spread across the whole window, sat at 1.0-1.1.
Nothing else in the diagnostics moves when the prefix freezes.

The block reports the profile plus two numbers to act on: **prefix renewal**,
the mean over the first five bins, and the **renewal gradient**, last bin minus
first — near 0 when renewal is uniform in time, large when it is concentrated
late. It also reports the **ancestor-sampling acceptance rate** beside them,
because a near-zero rate says the ancestor splice is contributing nothing to
renewal and the profile will not improve on its own.

The per-sweep columns are in each chain's `trace.tsv`: `trajectory_renewal` for
the aggregate, and one column per bin from `renewal_b0` to `renewal_b9`. Bin `b`
is the fraction of the series from `b/10` to `(b+1)/10` — a fixed tenth of the
substep index, so profiles compare across models, particle counts and step
sizes. (Measured: halving `dt` moved the gradient 0.141 → 0.172, which is the
invariance the fixed bins were chosen for.)

**Two things the gradient cannot do, worth knowing before you act on it.**

_It does not tell you why renewal is concentrated late._ It reads only the two
end bins, and two different shapes produce a large value: an early region flat
and near zero followed by a step — the coalesced genealogy above — or a smooth
monotone ramp, which is the ordinary finite coalescence depth of a long series
and can carry a perfectly respectable prefix. camdl's own `polio_afp_es` fixture
reports a gradient of 0.93 on a `0.06 → 0.31 → 0.53 → … → 0.99` ramp with a
prefix of 0.449. Firing there is defensible; concluding "the early path is
frozen" from it is not. Look at the profile and see which shape you have.

_It can miss a frozen prefix on a short series._ Where the last bin is itself
low, the gradient is bounded below the warning threshold however frozen the
early path is. A measured 60-step SIRS run has `b0 = 0.000` and a prefix of
0.001 — completely frozen — and a gradient of only 0.402, because its last bin
is 0.402 too. That run is caught by the aggregate rule (`trajectory_renewal`
below 0.10) and not by the gradient. **The two readings cover different cases;
read both.**

**Renewal says whether the path moved; it cannot say whether the chains agree
about the states that did not move.** A frozen prefix is benign if the data pin
those states — every chain would hold the same values, and the sampler's failure
to revisit them costs nothing. It is not benign if each chain is holding one
draw from a wide posterior, because then θ is being conditioned on a prefix the
sampler never revisits and the parameter R̂ is measuring agreement between chains
that were never in the same place. The **latent-path convergence** block
(printed directly under the renewal profile when a stage has at least two chains
and four saved paths per chain; `latent_convergence` in `pgas_summary.json`,
per-cell table `latent_convergence.tsv`) answers that question. It is also
recomputed from `chain_N/trajectories.tsv` by `camdl fit summary <fit>` for any
finished PGAS stage, so a fit that ran before the block existed gets it without
a re-run (the table is written once if the stage has none). It runs every
(state, substep) cell of the saved paths — compartments, `flow_*`, `inc_*` —
through the same rank-normalised R̂ the parameter table uses, and classifies
each: **constant** (the pooled draws never moved: structurally zero, or pinned),
**frozen-disagree** (every chain internally constant _and_ the chains differ:
one draw per chain, R̂ undefined), or **mixed** (R̂ and ESS computed). The rows
are the same tenths as the renewal profile, so they read together. Measured on a
three-province Ebola model (42 compartments, 12 observation streams, 19,200
particles, 4 chains, 100 sweeps, all 100 paths saved):

```text
path renewal:
  bin         b0    b1    b2    b3    b4    b5    b6    b7    b8    b9
  renewal  0.003 0.003 0.005 0.007 0.007 0.012 0.020 0.036 0.262 0.926

latent-path convergence (4 chains × 100 saved paths, 106 substeps × 99 columns):
  bin                  b0     b1     b2     b3     b4     b5     b6     b7     b8     b9
  frozen-disagree   0.191  0.163  0.147  0.064  0.034  0.014  0.005  0.000  0.000  0.000
  constant          0.176  0.113  0.088  0.082  0.071  0.060  0.050  0.034  0.015  0.003
  chains frozen     0.808  0.796  0.790  0.620  0.570  0.430  0.239  0.050  0.018  0.011
  R̂ max (mixed)    22.782 22.782 22.782 23.895 26.372 25.828  9.983  6.346  3.569  2.658
  ESS min (mixed)       4      4      4      4      4      4      4      5      5      5
  chains agree from substep 73 of 106: before it some state is one draw per chain
```

Read the two blocks as one. Renewal 0.003 in `b0` says the first tenth of the
path changed in a third of a percent of sweeps; the latent rows say what that
cost. A fifth of the early cells are one draw per chain outright, and the
`chains frozen` row says the "mixed" majority is barely better: over the
non-constant cells of the first tenth, 81% of chains never moved. That row
exists because a cell counts as mixed the moment one chain moves once, and
`frac_mixed` alone would read 0.63 there. R̂ up to 23 with an ESS of 4 on 400
saved paths is four chains at four different values. The table has the concrete
row: the exposed count in one province at day 5 is held at 198, 153 and 152 in
three chains for the whole run (`n_frozen_chains` = 3), and the fourth chain
moved exactly once, at sweep 7, from 187 to 110; between-chain SD 34 against
within-chain SD 10. That is the row to look at before reading the parameter R̂ of
anything whose likelihood lives in the early path. `agree_from` is the horizon:
after substep 73 no state is one draw per chain, so the chains' paths mix over
the last third of the series and whatever the tail identifies — the
observation-model parameters, here — is being estimated from a posterior the
chains actually share.

One signature is worth knowing. The same R̂ (22.782) recurs at dozens of cells
over the first thirty substeps. That is not a coincidence: rank-normalised R̂ is
computed from the ranks of the draws, not their values, and when the early path
renews at all it renews as a block — an accepted ancestor splice shifts every
later state of that sweep's path at once — so every state in the prefix shares
one pattern of "which chain changed, at which sweep" and hence one R̂. Here it is
the single sweep-7 splice in one chain, visible at every one of those cells. A
run of identical R̂ values along the series is the whole-prefix renewal event
seen from the state side.

What it changes: the numbers are reported without a verdict, because the
threshold question for latent R̂ is open, but the reading is the same as for
parameters — an R̂ far above 1 with an ESS near the chain count is not a
posterior. A frozen-disagree fraction that falls to zero only late, with the
renewal profile flat over the same bins, is the particle-limited case above:
raise the particle count and re-read both blocks. Frozen-disagree cells that
persist at a particle count where renewal has recovered point at the model
instead — a state the data do not constrain, held wherever the initial filter
draw put it. The ESS here is over the saved paths (`n_trajectories`), not every
sweep; save more paths if it is the number you need.

**Filter ESS: the one number renewal and R̂ cannot see.** Renewal and the
latent-path block read the _output_ of each sweep. The filter ESS block
(`filter_ess` in `pgas_summary.json`, printed under the two above, per-chain
table `filter_ess.tsv`) reads the sweep's _inside_: at every observation, the
effective sample size $(\sum w)^2 / \sum w^2$ of the particle weights the
resample draws from, pooled as a mean and a minimum over the retained sweeps of
every chain. The `trace.tsv` columns `collapsed_windows` and `min_alive` count
particles whose weight is _finite_, and a finite weight can be negligible. On a
19,200-particle single-province Ebola fit every sweep reported `min_alive`
between 4,357 and 19,166 — nothing collapsed, on that reading — while at one
observation, a re-issued cumulative count that had been floored to zero, the
filter's ESS was between 2 and 3 in every sweep. A handful of particles could
reach a zero; every other particle scored a density near $e^{-25}$; the resample
copied those few into every slot, and the whole path through that day was one of
two or three draws, sweep after sweep. The renewal profile and the parameter R̂
both read as healthy, because they were: the sampler was mixing fine over a
likelihood that had a hole in it.

The block reports the minimum, 10% quantile and median of the per-observation
mean ESS, and lists the **starved** observations — mean ESS below 1% of the
particle count — worst first, with their times. That threshold is deliberate:
the bootstrap filter's own bail floor (an ESS of 2) would have passed 2.2, and
the prequential collapse rule (a tenth of N) fires on the healthy peak of an
epidemic, where an ESS of 4% of N is ordinary. One percent separates "small"
from "a handful". When something is starved, go to the data row at that time
before touching the model: a starved observation is usually a value the model
cannot reach — a floored re-issue, a decimal-shifted count, a stream that
changed definition — and the fix is to the data or to the observation model's
dispersion, not to the particle count. Add particles only if a 4× re-run moves
the mean ESS at that observation by about 4× too; if it stays at a handful, no N
reaches it. The per-sweep minimum and where it fell are also in `trace.tsv`
(`min_ess`, `min_ess_t`), so a single bad sweep can be told apart from a bad
observation.

**If you are running PGAS, this is your bullet — do not read past it into (b).**
The fix for a starved CSMC is particles, and it looks nothing like the fix for
geometry.

For PMMH and IF2, first ask whether the noise is even fixable: scale particles
at one fixed θ and watch the loglik standard deviation.

```bash
camdl pfilter model.camdl --params theta.toml --data cases.tsv \
    --replicates 20 --particles 1000 --output ll_1k.tsv
# repeat at --particles 4000, 16000 and compare the reported loglik ± SD
```

If the SD falls like $1/\sqrt{N}$, more particles help. If it **plateaus**, no
particle count saves you — this happens when many observation streams are
observed at the same time (high _effective observation dimension_), where the
bootstrap filter is structurally inadequate. `camdl pfilter --pf-health`
measures this directly (the Snyder et al. 2008 $\exp(\tau^2/2)$ implied-N
estimate); see [`camdl docs inference`](inference.md) (the `--pf-health`
section). The fix there is a different method, not brute-force N.

For PGAS the equivalent probe is one re-run at roughly 4× the particles,
comparing R̂ and the renewal profile. Measured on a national Ebola PGAS fit —
same model, same data, same config, same 8,000 sweeps, **only `particles`
differs** — with per-parameter R̂:

| `particles` | chains | `trajectory_renewal` | `r_eff` | `tau` | `q_comm` | `gamma` | `rho` | `rho_lab` |
| ----------- | ------ | -------------------- | ------- | ----- | -------- | ------- | ----- | --------- |
| 1,200       | 6      | 0.591                | 1.33    | 2.52  | 2.64     | 1.81    | 1.18  | 1.19      |
| 4,800       | 7      | 0.707                | 1.02    | 1.46  | 1.32     | 1.37    | 1.03  | 1.03      |

Every parameter improved, acceptance and divergence counts became healthy, and
one further chain cleared the initialisation check (hence 6 → 7 from an
unchanged config). Nothing about the model changed. Before that re-run the team
had spent a day reading these R̂ values as (b) and was about to reparameterise.

Read the probe on the profile and it says more. A second matched probe on one
model and dataset, 200 sweeps, changing only the particle count:

| `particles` | b0   | b1   | b2   | b3   | b4   | b5   | b6   | b7   | b8   | b9   | `as_accept` |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| 4,800       | 0.07 | 0.07 | 0.07 | 0.08 | 0.08 | 0.09 | 0.51 | 0.77 | 0.82 | 0.98 | 0.0160      |
| 19,200      | 0.21 | 0.23 | 0.23 | 0.23 | 0.24 | 0.24 | 0.67 | 0.93 | 0.94 | 0.99 | 0.0157      |

Four times the particles roughly triples renewal over the first six bins, so the
frozen prefix is particle-limited rather than geometric — the profile answers
"will more particles help" that the aggregate cannot. The ancestor-sampling
acceptance rate meanwhile does not move (0.0160 → 0.0157): ancestor sampling
contributes essentially nothing on this model and more particles do not change
that, which is what to expect for an integer compartment state whose ancestor
weight is sharply peaked and often exactly zero on support grounds. That is only
visible because the two are reported side by side.

**A low acceptance rate has two causes, and four columns tell them apart.**
Ancestor sampling is the step that re-attaches the reference trajectory's prefix
onto another particle's history — the mechanism whose effect the renewal profile
measures. At each substep it computes one weight per particle (Lindsten, Jordan
& Schön 2014, Eq. 17), screens out candidates whose splice would strand the
reference's later recorded flows, and draws the new ancestor from a categorical
distribution over the survivors. Four `trace.tsv` columns describe that draw:

| column               | what it holds                                                                 |
| -------------------- | ----------------------------------------------------------------------------- |
| `as_finite_frac`     | fraction of the ensemble carrying a finite ancestor weight, before the screen |
| `as_admissible_frac` | the same fraction after the screen                                            |
| `as_ess_pre`         | effective sample size of those weights before the screen, **in particles**    |
| `as_ess_post`        | the same after the screen, in particles                                       |

The first two count candidates; the last two count choices, and on a real fit
the two come apart by orders of magnitude. Effective sample size here is
$(\sum w)^2 / \sum w^2$ over the ancestor weights — the number of
equally-weighted candidates that would give the same draw. At
`as_admissible_frac = 0.24` on 4,800 particles roughly 1,150 candidates survive
the screen; if one of them carries almost all the weight, the categorical picks
it nearly every time, the effective number is 1, and the ancestor move renews
nothing however many candidates are nominally admissible.

**The two ESS columns are particle counts, while the two beside them are
fractions.** Read `as_ess_post` against `as_admissible_frac × particles` — the
candidates it is an effective count _of_. An ESS of 3 out of 4 candidates is
unremarkable; an ESS of 3 out of 1,150 is a categorical with one real choice.
Both read `NA`, never `0`, on a sweep that ran no ancestor-sampling step: no
data is not a measured collapse.

Both sides of the screen are reported because the screen can _raise_ the ESS
while lowering the count. Eight particles, normalised ancestor weights:

```text
0.90  0.04  0.02  0.02  0.01  0.01   -inf  -inf    6 finite, ESS 1.23
 --   0.40  0.20  0.20  0.10  0.10                 5 admissible, ESS 3.85
```

The count fell and the ESS more than tripled: the screen removed a dominant
candidate whose splice was infeasible and which had been monopolising the draw.
By the counts alone the screen looks harmful. Hence the four readings:

| `as_ess_pre` | `as_ess_post` | what it says                                                                                                                                                                                     |
| ------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| high         | high          | the weights are spread and the screen is neutral — a low acceptance rate is the exact suffix ratio rejecting real choices, so the proposal is fine and the update itself (blocking) is the lever |
| high         | low           | the screen is concentrating the draw; the bottleneck is the backward-feasibility test, not the density                                                                                           |
| low          | low           | the density itself piles onto one ancestor; the ancestor proposal needs improving, and no change to the screen will help                                                                         |
| low          | high          | the screen removed a dominant infeasible candidate — working as intended                                                                                                                         |

Without them, a low acceptance rate is compatible with both "the move has good
candidates and the Metropolis step rejects them" and "the move never had a
choice", which have different fixes and very different costs. The medians over
the retained sweeps are in the `path_renewal` block of `pgas_summary.json` and
printed at the end of the stage beside the acceptance rate; the per-sweep values
are the `trace.tsv` columns above, written as the run proceeds.

**Read the acceptance rate where it happened, not as one number.** The ancestor
move can behave completely differently at the start of the series and at the
end, and one rate averaged over the whole sweep hides that. `trace.tsv` carries
`as_accept_b0 … as_accept_b9` — the acceptance rate among the Metropolis steps
that ran in each tenth of the substep series, on the **same ten bins** as
`renewal_b0 … renewal_b9`, so the two rows are read against each other. The
end-of-stage block prints them as a pair — here the two rows of a 16-sweep,
32-particle, 80-substep SIR test fixture, small enough that single bins are
noisy, shown for the layout:

```text
bin           b0    b1    b2    b3    b4    b5    b6    b7    b8    b9
renewal    0.477 0.703 0.562 0.562 0.680 0.797 0.867 0.820 0.875 0.922
as accept  0.500 0.385 0.167 0.000 0.062 0.000 0.000 0.375 0.600 0.875
```

A bin reads `NA` where no ancestor move was ever proposed in it. That is not an
acceptance rate of zero, and the difference is the point: "the move was never
offered here" and "it was offered here and always refused" send you to different
places.

**A profile that falls toward `b0`** — high over the last tenths of the series,
near zero over the first — says the splice gets harder the further back it is
attempted. That is the expected mechanism: re-attaching the reference path at an
early substep requires everything recorded _after_ that point to stay plausible
under the new ancestor, and there is more of it the earlier you go. The remedy
is on the proposal and the update, not on the ratio: more particles (which
measurably raised the early renewal bins in the probe above), or a proposal that
uses information from the suffix it will have to keep. The single rate such a
profile averages to reads as a uniformly mediocre move and sends you looking in
the wrong place.

**A flat profile** — the same rate in every bin, whatever that rate is — says
the opposite: position in the series is not what determines whether a splice
lands. The proposal is mismatched to the target uniformly in time, so the lever
is the proposal density or the structure of the update (blocking), and adding
particles will not change the shape. A flat profile at a _low_ rate with a
healthy `as_ess_post` is the "restructure the update" cell of the table above,
arrived at from the other direction.

The two shapes are what the sweep-level rate cannot separate: a run accepting
nothing before the midpoint and everything after it, and a run accepting half of
everything everywhere, report the identical scalar.

**When you re-run at more particles, compare the profile and the aggregate — not
the gradient.** The gradient describes a shape, and raising the particle count
can steepen that shape even as the sampler improves. Measured on one model
family with model, data and sweeps held fixed, raising `particles` 100 → 400 →
1600 moved the gradient 0.812 → 0.857 → 0.899 while the aggregate improved. The
verdict does not change — the warning fires at all three — but a user who reads
the gradient as a progress bar will conclude the re-run made things worse.

**The ordering lesson, which is the generalisable part: establish that the
instrument is adequate before concluding anything about the model or its
geometry.** An under-resourced filter produces symptoms that mimic
non-identifiability — poor R̂ on exactly the parameters a weakly-identified
direction would spoil. Rule out (a) first; it costs one re-run.

**(b) Geometry (ridges, flat or stiff directions)** stalls gradient-based
PGAS-NUTS, which is a different problem with a different fix (reparameterize,
tighten priors, or add identifying data). Conclude (b) only once (a) is ruled
out.

**Crucially: PGAS-NUTS is immune to the PF marginal noise of (a).** It runs on
the _smooth complete-data conditional_ likelihood — it conditions on a sampled
latent trajectory rather than marginalizing it out with a noisy filter — so
"PMMH is dead on this problem" does **not** imply "PGAS is dead." If marginal
noise is killing PMMH/IF2, PGAS is often still viable. (See the
marginalize-vs-condition contrast in [`camdl docs inference`](inference.md).)

**Immune to the noise is not immune to the particle count.** PGAS still runs a
CSMC sweep, and that sweep still needs particles to renew the reference
trajectory — the mixing bullet in (a). "PGAS does not eat the marginal
likelihood" is a reason not to fear `loglik_se`; it is not a reason to run PGAS
at a particle count chosen for a cheap smoke test.

**(c) When the filter can't be rescued in place — scaffold with `ode + mh`, and
read the dying filter as a signal.** If the particle filter is dying (ESS
collapsing, particles going extinct) and you are already at a high particle
count or capped, stop adding particles and switch to the **`ode + mh`** backend.
It scores the _deterministic_ marginal likelihood directly — there is no
particle filter, so there is nothing to degenerate — and it explores fast,
giving a working posterior even where the stochastic filter is hopeless. Treat
it as a triage scaffold, not a compromise.

And read an _un-fixable_ dying filter as evidence about the model. If more
particles do not help **and** the filter is healthier on synthetic data than on
the real data (§1), the model often cannot produce data consistent with what you
observed — so the filter cannot keep particles alive near the observations. That
is **observation-model misspecification**, distinct from the structural
obs-dimension collapse in (a): the cure is to fix the observation process, not
the sampler. The productive sequence is to get a clean `ode + mh` fit first, let
its residuals, calibration (§5), and compensation signatures (§6) point at the
observation-model gap, fix the observation process there, and only then return
to the stochastic fit — which is often viable again once the model is
well-specified. This is the exact arc of a 14-stream spatial model whose
bootstrap PF was dead on the joint streams: `ode + mh` converged in minutes; its
single shared dispersion `k` pinned at the floor with overconfident intervals (a
calibration tell, §5/§6); a partially-pooled per-stratum dispersion fixed the
observation model; and the better-specified, non-centered model then ran under
PGAS.

### Choosing the backend by regime, up front

Scale selects the tool — you can often skip the failure-mode tree by matching
the backend to the regime before you start:

- **Large-population, near-deterministic systems → `ode` (`mh` / `nl-sbplx`).**
  Demographic noise is $O(1/\sqrt{N})$, so the deterministic skeleton ≈ the
  stochastic mean; and the over-constrained latent makes PF/PGAS choke — the
  same near-determinism that makes the ODE _accurate_ is what starves the
  filter.
- **Small-population / fade-out regimes → `chain_binomial` (`PMMH` / `PGAS`).**
  Stochasticity is real and the latent is not over-constrained.
- **Many simultaneous observation streams → not the bootstrap PF, at any scale**
  (the (a) collapse).

### Diagnosing and fixing a geometry stall (b)

When PGAS-NUTS stalls on geometry rather than noise, three moves, in order.

**First, know what R̂ can and cannot answer.** R̂ measures whether the chains
_agree_; it does not measure whether the data _determine_ a parameter. On a
genuinely flat direction, better mixing improves R̂ too — the chains explore the
flat direction more fully and agree on the same wide distribution — so a falling
R̂ is not evidence of identification, and a high R̂ is not proof of
non-identification. The instrument for identifiability is **prior-to-posterior
shrinkage** — posterior width divided by prior width — because unlike R̂ it does
not route through between-chain agreement at all.

**But shrinkage still has to be _estimated_, and that needs adequate ESS.** A
pooled 90% interval computed from badly-mixed chains describes where the chains
happened to sit, not the posterior: it inflates when mixing is poor and tightens
when mixing improves, exactly as R̂ does. Both statistics moved together in the
Ebola fit of §3(a) when the particle count rose — `tau`'s R̂ went 2.52 → 1.46 and
its 90% posterior width 0.294 → 0.204, from 79% to 55% of the prior's 90% width
of 0.371. At the per-chain effective sample sizes those runs delivered (order
5), neither number is trustworthy; 55% of prior is the best available reading of
how much the data constrain `tau`, and it is still not a reliable one. So: reach
for shrinkage rather than an R̂ contrast when the question is identifiability,
quote the ESS beside it, and re-read §3(a) before calling anything geometry.

**Name the bad direction with R̂.** If one parameter's R̂ won't converge but the
fit reproduces the data fine, suspect a _non-identified combination_, not a bad
parameter. Look at the posterior correlation matrix; if two parameters are ≈ ±1
correlated, compute R̂ on the **identified combination** (e.g. the product
$\rho \cdot D_{50}$) versus the **orthogonal** direction (the ratio): the
combination converges (R̂ ≈ 1), the sloppy direction does not. That decomposition
names the direction the chains disagree on, in one line. Confirm it with
shrinkage before reporting it as non-identifiability — an R̂ contrast on its own
can be nothing more than a contrast between a quantity the sampler happened to
mix on and one it did not. (A sloppy ridge in the sense of Gutenkunst et al.
2007; gh#263 proposes to automate this as a post-fit report.)

**Don't be fooled by warm-up.** R̂ ≈ 1.1 with a tail that _looks_ converged is
usually just too-short burn-in inflating the statistic. Recompute split-R̂
(Vehtari et al. 2021) after discarding the warm-up transient — 1.12 → ~1.00 from
exactly this is common. (gh#262's warm-up-aware R̂ surfaces it live.)

**Reparameterize the funnel.** The highest-yield fix for a hierarchical/pooled
parameter is the **non-centered** form. A centered
$k_p \sim \mathrm{LogNormal}(\mu, \tau)$ funnels — chains are lost and R̂ is
hopeless under gradient-free MH. Write it non-centered instead:
$k_{\text{raw}} \sim \mathrm{Normal}(0, 1)$ and
$k = \exp(\mu + \tau \cdot k_{\text{raw}})$ via a `let` (Betancourt & Girolami
2015). Bonus: because the hierarchy now lives in a `let` rather than a _declared
hierarchical prior_, the non-centered form also sidesteps PGAS's
hierarchical-prior gate (gh#175) — so it is what lets a pooled model run under
PGAS at all.

**Reseed a frozen sampler.** If PGAS-NUTS freezes from a cold start (step size
collapsing to ~$10^{-4}$), seed it at the `nl-sbplx` / `mh`-on-`ode` MLE: clean
latent trajectories let NUTS navigate locally where it stalled from a bad start.
(This is the IF2→PGAS warm-start in [`camdl docs inference`](inference.md),
extended to the deterministic backends.)

## 4. Pinning parameters helps geometry, not PF noise

Fixing or pinning parameters (`--fixed name=value`) reduces dimension and can
unstick the **geometry** problem (b) — fewer ridges to climb. It does
**nothing** for the under-resourced filter of (a): the ESS at a given θ does not
depend on how many parameters are free, and neither does whether CSMC renews the
reference trajectory. If your problem is the filter, pinning parameters will
feel like it should help and won't. Attack the problem you actually have.

## 5. Use the right "predicted value" for the diagnostic

Three different predictions answer three different questions; using the wrong
one hides the very misfit you're hunting.

- **Free-forward (unconditional posterior-predictive)** —
  `camdl fit predict --fit fit.toml --horizon free_forward`. Replays the fitted
  model from the start, never re-anchored to data, sampling `y_rep` per
  posterior draw. Exposes _generative_ misspecification: can the model, run
  forward on its own, produce data like the observations? A drifting transition
  model produces a band that blows up over time.
- **One-step-ahead** — `camdl fit predict --fit fit.toml --horizon one_step`.
  The honest short-horizon forecast `p(y_t | y_{1:t-1})`: re-conditions on the
  data at every step, so it stays tight **iff** the filter can track the data.
  The right tool for _timing_ questions (does the model anticipate each next
  observation?). Chain-binomial only. A model can pass the free-forward check
  (wide enough to contain the data) yet fail one-step (can't predict next week),
  and vice versa — which is why both are worth emitting and carry a typed
  `horizon` column so neither is read as the other.
- **Conditioned / smoothed path** — `camdl pfilter --save-paths`. This is pulled
  toward the data by construction and will track it even for a misspecified
  model. It **cheats** for the purpose of model-checking: a smoothed ribbon that
  hugs the data is not evidence the model is right.

`fit predict` writes both predictive horizons (stacked in
`predictive/<stream>.tsv`, keyed by the `horizon` column) and the observed
series (`observed/<stream>.tsv`) under the run directory, ready to join on
`(time, <dims>)` and plot — one facet per stratum. Read the predictive band as a
**quantile ribbon** (`q05…q95`), never a mean: averaging stochastic replicates
with jittered epidemic take-offs smears the peak later and lower than any single
run. The divergence between the free-forward ribbon and the smoothed ribbon _is_
the diagnostic — see the "unconditional vs smoothing" plot in
[`camdl docs inference`](inference.md).

Separately from _where_ a prediction sits, check whether its **intervals** are
trustworthy — the calibration question the three views above don't answer. Two
standard checks: the **PIT histogram** (the probability-integral transform of
each observation under its predictive distribution — U-shaped ⇒ intervals too
narrow / overconfident, domed ⇒ too wide; Gneiting et al. 2007) and
**nominal-vs-empirical coverage** (does the 90% interval actually contain ~90%
of held-out points?). This is exactly what separated the overconfident
shared-`k` fit (U-shaped PIT) from the well-calibrated hierarchical-`k` fit in
§3(c).

**Then read the surprise table before trusting any of the aggregates.** The
elpd, the mean CRPS and the PIT coverage are all sums or averages over
observations, and an average hides which terms carry it. Wherever camdl prints
the elpd — the `pfilter` stage of `fit run`, `camdl pfilter --save-prequential`
— it prints beneath it the five worst-scored observations of the trace, worst
first, each with its share of the elpd, its PIT, and the filter's ESS at that
step. The Ebola case that earned this table: an elpd of a few hundred nats over
103 days read as ordinary, and so did the CRPS and the coverage; one day — a
cumulative count re-issued and floored to zero — carried a log score of −26.6 on
its own, a PIT of 0 and a filter ESS of 3, and stood only as one row of
`prequential.tsv` that nobody opened. The reading is by shape. Shares that are
flat at a few percent each say no observation dominates and the aggregates mean
what they say. One row an order of magnitude below the rest, with `pit` pinned
at 0 or 1 and `ess` in the single digits, is one observation the filter could
not explain, and the first move is the data row at that `t`, not the model —
that is also the observation `filter_ess` (§3) will have flagged if you ran
PGAS. Several bad rows sharing a stream point at that stream's observation
model; several sharing a window point at the transition model over that window.

## 6. Read the MLE for "compensation" signatures

A point estimate can be a symptom rather than an answer. Watch for:

- a parameter **pinned at a bound** (survey's bound-clustering warning and the
  IF2 chain-agreement ranges surface this);
- an **unphysical** value — $R_0 < 1$ for an outbreak that visibly grew,
  overdispersion slammed to its maximum, a reporting rate at 0 or 1.

These are the optimizer contorting one parameter to absorb a structural misfit
elsewhere. When you see one, the estimate is telling you the model is wrong, not
giving you the answer — go back to §1.

A common specific case: a single negative-binomial dispersion `k` **pinned at
its floor (maximum overdispersion) across all strata**. The data are burstier
than one shared `k` can represent, and the fix is not a looser bound but more
structure — a partially-pooled (hierarchical) per-stratum dispersion, fit
non-centered (§3(b)). A floor-pinned shared `k` and the overconfident intervals
it produces (§5) are the paired tell that sent the 14-stream model in §3(c) to a
hierarchical observation model.

## 7. The meta-lesson: a fighting sampler is doing model-checking

A framework whose particle filter degenerates and whose sampler stalls on a
misspecified model is, in effect, performing model criticism for you. That
pushback is a **feature**, not a defect to tune away. When the sampler fights
you, suspect the model before the sampler — §1 is how you confirm it.

## camdl-specific gotchas

Concrete things that trip up real fits, verified against the current code:

- **PGAS (and PMMH) require a prior on every estimated parameter.** A parameter
  with no `~` in the model and no prior in the fit toml is a **hard error** that
  names the offending parameters and the three remedies — not a silent fallback.
  Declare priors via `~` in the model, in `[estimate.<param>].prior`, or opt
  into flat priors explicitly. (IF2 and the NLopt optimizers ignore priors.)
- **The ODE backend does MLE via `nl-sbplx` / `nl-bobyqa`, and Bayesian via
  `mh`** (Metropolis-Hastings on the deterministic marginal likelihood). PGAS
  and PMMH are **chain-binomial only** — they need stochastic process variance
  the ODE backend doesn't have, so asking for them on `ode` is a hard error that
  points you at `mh`. Run `camdl fit methods` for the current matrix.
- **The PGAS trace's loglik column is named `log_complete_data_ll`** — the
  complete-data conditional value (it conditions on the full sampled latent
  path: initial + transition + observation density), a large-negative number,
  **not** a marginal/PF likelihood. (PMMH and `mh` report the marginal in their
  `log_likelihood` column.) Don't compare PGAS's `log_complete_data_ll` to a
  `camdl pfilter` loglik — they differ by orders of magnitude.
- **The bootstrap particle filter degenerates with many simultaneous observation
  streams.** Per-stream likelihoods multiply into one weight, so high
  observation dimension collapses ESS and more particles only buy
  $\exp(\tau^2/2)$ headroom. Measure it with `camdl pfilter --pf-health` before
  scaling N (§3a).
