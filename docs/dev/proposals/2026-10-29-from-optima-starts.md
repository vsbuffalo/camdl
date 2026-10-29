---
date: 2026-10-29
status: proposed — every design question is decided in §12 with a recommendation and a confidence; awaiting maintainer review
area: chain starts (`fit/starts.rs`, `fit/chain_starts.rs`) / fit summary / run identity
related:
  - 2026-09-08-workflow-first-fit-config.md (removed `[stages]` and `init_mle`; §1.3 and §3.4 are what this must not undo)
  - 2026-10-29-support-keyed-transforms.md (D3, starts on a bound; the revision on `feat/support-transforms` is the one this follows)
---

# Start each sampler chain at a distinct good optimum of a stored IF2 fit

Code references are against `origin/main` at `0abe59f8` unless another ref is
named. A claim marked _verified_ was read at the cited line or produced by the
command shown. _Inferred_ means reasoned from the code and not executed. Nothing
in this proposal has been implemented or run; every number quoted is from a
committed artifact named beside it.

## 1. The rule, the workflow, and the one claim of the earlier proposal it revises

This proposal adds one sourced chain-start rule:

```toml
[method]
algorithm = "pgas"
starts = { from_optima = "@scout" }
```

or, on the command line, `--starts from_optima=@scout`. Each chain of a
posterior sampler (PGAS — particle Gibbs with ancestor sampling — and PMMH —
particle marginal Metropolis–Hastings) begins at the end point of a different
IF2 chain of the stored fit `@scout`. IF2 is iterated filtering (Ionides,
Nguyen, Atchadé, Stoev and King 2015, _PNAS_ 112:719–724), camdl's
maximum-likelihood method; each of its chains climbs the likelihood from its own
start. Only chains whose likelihood at their end point is close to the best
chain's are used ("good" chains, §7.4). If the source has fewer good chains than
the sampler has chains, the run is refused with a message saying how many IF2
chains to run instead. No IF2 end point is used twice.

The workflow is four commands and needs no pipeline machinery:

```bash
camdl fit new --from fit.toml fit-if2.toml          # copy the problem; edit [method] to IF2
camdl fit run fit-if2.toml --starts from_prior --label scout
camdl fit summary fit-if2.toml                      # shows which IF2 chains are good
camdl fit run fit.toml --starts from_optima=@scout
```

The syntax of each command is _verified_: `fit new --from <src> <dest>`
(`rust/crates/cli/src/args/mod.rs:1458-1471`, `FitNewArgs` at `:1464`),
`--starts` and `--label` on `fit run` (`args/mod.rs:681`, `:780`), a `fit.toml`
path as a fit handle (`fit/handle.rs:7`). `fit new` copies the source file and
inserts a `[provenance]` block (`fit/mod.rs:2476-2510`); it does not change
`[method]`, so the user sets `algorithm = "if2"` and the chain count in
`fit-if2.toml` by hand. §8 gives the recommended IF2 chain count.

**This changes one conclusion of the workflow-first proposal.** That proposal
deleted in-file chaining and wrote that "the feature's sole justified use —
priming PGAS with an IF2 point — is no longer needed in practice and is harmful
when used" (`2026-09-08-workflow-first-fit-config.md`, §2). The second half was
measured and stands: starting every chain at one IF2 point makes R̂ (the
potential scale reduction factor, which compares between-chain to within-chain
variance) agree by construction. The first half is now contradicted by the
spline recovery probe of §3: on that fixture, prior-draw starts left one PGAS
chain at a parameter value whose marginal log-likelihood is 1,373 nats below the
other chains', and IF2-assisted starts removed that in every case tried. This
proposal is the version of IF2 priming that does not collapse the chains (§5),
and its merge is conditional on a measurement on three fixtures (§9), not on the
one probe.

What the rule does not change: the target density. A start decides where a chain
begins, never what it samples, so every failure of this rule is a convergence
failure — a chain that has not reached or not explored the posterior — and not a
different estimand. That is why the measurement in §9 is about convergence and
agreement with a reference, and why a mismatch between the source's problem and
the target's is a notice rather than a refusal (D12).

## 2. Terms

- **Source** — the stored IF2 fit the rule reads (`@scout`). **Target** — the
  sampler run that reads it.
- **End point of an IF2 chain** — the chain's final-iteration swarm mean, which
  IF2 re-scores with a separate high-particle filter (the _clean evaluation_)
  and writes, one row per chain, to `chain_evaluations.tsv` (§4.3).
- **Clean-eval log-likelihood ℓ_c and its standard error se_c** — chain c's
  re-scored log-likelihood and the Monte Carlo standard error of that score,
  both from `chain_evaluations.tsv`.
- **Decibans (dB)** — a log-likelihood difference in base-10 tenths: Δ dB = Δ
  nats × 10 / ln 10 (`NATS_TO_DB`, `rust/crates/cli/src/evidence.rs:63`). The
  default IF2 threshold of 30 dB is 6.91 nats, a likelihood ratio of about
  1,000.
- **`decibans_thresh`** — the IF2 configuration key (`[method.gate]`, default
  30.0, `fit/config_v2.rs:1860-1864`) used by the IF2 chain-spread check (§4.2).
- **Good chain** — a source chain whose end point passes §7.4's test.
- **Trapped chain** — a sampler chain whose parameter value at the end of its
  run has a marginal log-likelihood, log p(y | θ), more than the decibans
  threshold below the best chain's in the same run. Defined fully in §9.3; the
  probe's example is a gap of 1,373 nats.
- **Bulk-ESS and tail-ESS** — effective sample sizes for the posterior centre
  and for its 5% and 95% quantiles (Vehtari, Gelman, Simpson, Carpenter and
  Bürkner 2021, _Bayesian Analysis_ 16:667–718), as camdl computes them since
  gh#84.
- **Spread / point** — the two kinds of start rule today: one independent draw
  per chain, or every chain at one vector (`fit/starts.rs:47-54`). §7.1 adds a
  third kind, `optima`.

## 3. The evidence is one probe on one fixture, and it differs from this design in five ways

All files below are committed on `feat/bsplines-s2` (head `3169522f`) and read
there. The fixture is a SEIR model whose transmission rate β(t) is a five-knot
B-spline, written in two forms: `explicit` (five free coefficients `b1..b5`) and
`smooth` (a penalised form with level `a`, slope `b`, modes `z_m1..z_m3` and a
smoothness scale τ). Four synthetic datasets (data seeds 23, 31, 47, 59). Every
PGAS run below has 4 chains, 64 particles, 1,000 adaptation and 3,000 kept
sweeps, except the one-chain restarts of the first table (500 adaptation and
1,000 kept sweeps).

**Prior-draw starts left one chain at a poor parameter value.** For the two runs
whose chains disagreed most, `scripts/diagnose_stuck_chains.py` scored each
chain's last kept θ with `camdl pfilter` (2,000 particles, five seeds) and then
restarted PGAS at that θ alone. From
`tests/recovery/cases/spline_seir/results/diagnosis/stuck_chains.tsv`:

| form     | seed | chain | marginal log-lik (mean ± sd) | complete-data log-lik after restart at that θ |
| -------- | ---- | ----- | ---------------------------- | --------------------------------------------- |
| smooth   | 31   | 0     | −514.9 ± 0.09                | −1,736.3                                      |
| smooth   | 31   | 1     | −514.0 ± 0.08                | −1,735.9                                      |
| smooth   | 31   | 2     | **−1,886.8 ± 5.17**          | **−3,386.9**                                  |
| smooth   | 31   | 3     | −517.0 ± 0.12                | −1,736.8                                      |
| explicit | 47   | 3     | −504.9 ± 0.19                | −1,706.1                                      |

Smooth seed 31 chain 2 sits 1,372.8 nats below the best chain, and restarting
the sampler at its θ with a fresh latent path did not move it out (its
complete-data log-likelihood stays near −3,387). It is stuck in θ, not in its
latent path. Explicit seed 47 chain 3 is the other kind: its θ scores like the
others' and it recovers once its path is re-drawn. Only the first kind is what
this rule addresses.

**IF2-assisted starts.** `scripts/diagnose_multistart.py` drew 16 prior points
per dataset, kept the first 8 the filter could score, ran one single-chain IF2
fit from each, re-scored each end point (2,000 particles, three seeds), kept
those within 5 nats of the best, merged those whose log β(t) curves differ by
less than 0.1, and started four independent single-chain PGAS fits at the
resulting solutions, cycling through them with small Gaussian jitter. From
`.../diagnosis/multistart/multistart_summary.tsv`, against the default
`from_prior` runs in `.../diagnosis/summary_default.tsv` (max over 41 window
times of log β(t)):

| form     | seed | from_prior R̂ | from_prior bulk-ESS | IF2-start R̂ | IF2-start bulk / tail-ESS | IF2 fits run / distinct good solutions |
| -------- | ---- | ------------ | ------------------- | ----------- | ------------------------- | -------------------------------------- |
| explicit | 23   | 1.266        | 12                  | 1.113       | 37 / 107                  | 8 / 2                                  |
| explicit | 31   | 1.180        | 17                  | 1.060       | 91 / 165                  | 8 / 1                                  |
| explicit | 47   | 1.499        | 8                   | 1.030       | 132 / 201                 | 8 / 1                                  |
| explicit | 59   | 1.233        | 13                  | 1.145       | 24 / 95                   | 8 / 2                                  |
| smooth   | 23   | 1.717        | 6                   | 1.085       | 55 / 124                  | 6 / 1                                  |
| smooth   | 31   | 2.945        | 5                   | 1.037       | 87 / 34                   | 5 / 1                                  |
| smooth   | 47   | 1.663        | 7                   | 1.026       | 117 / 43                  | 6 / 1                                  |
| smooth   | 59   | 1.719        | 6                   | 1.079       | 57 / 102                  | 7 / 2                                  |

No IF2-started chain was trapped in θ in any of the eight datasets. That is the
whole of the positive finding. Four qualifications attach to it, and each shapes
the design or the measurement:

1. **R̂ fell, but R̂ is not the evidence.** The same fixture started every chain
   at one prior-mean point (`starts = "single"`, `.../diagnosis/summary.tsv`,
   tuning `single_d8`) and got R̂ between 1.031 and 1.091 — as good as the
   IF2-assisted runs — because chains that begin together agree whether or not
   they have explored the posterior. A lower R̂ from concentrated starts is
   partly what concentration does. The measurement in §9 therefore counts
   trapped chains by their marginal likelihood and compares each arm's posterior
   to a reference, and reports R̂ only beside those.
2. **Not one IF2-assisted run is certified.** Four of eight R̂ values are at or
   above camdl's sampler threshold of 1.05 (`RHAT_CONVERGED_THRESHOLD`,
   `fit/method_result.rs:585`), and the smooth runs had 57 to 495 divergent
   transitions. The probe removed trapping; it did not produce a converged fit.
3. **The end points collapsed to one or two solutions.** In every dataset the 5
   to 8 IF2 fits gave one or two distinct good curves (the summary records fits
   run and distinct good solutions, not the count of good fits). A rule that
   demanded as many geometrically distinct optima as chains would have refused
   all eight datasets. This is why §7.5 counts distinct IF2 _chains_, not
   distinct _basins_.
4. **The probe is not this design.** It ran separate single-chain IF2 fits (this
   rule reads one multi-chain fit); it judged "good" by 5 nats on a fresh
   re-score (this rule uses the source's own `decibans_thresh`, 6.91 nats by
   default, on the clean evaluation, §7.4); it merged by curve distance (this
   rule does not merge, §7.5); it reused one or two solutions across four chains
   with jitter (this rule refuses rather than reuse, D5); and in the `smooth`
   form it fixed τ at its prior median in IF2 because IF2 refuses an estimated
   smoothness scale on that branch (`rust/crates/cli/tests/bspline_e2e.rs:344`
   on `feat/bsplines-s2`), which is the case D12 settles. The measurement must
   therefore be re-run with the rule as specified; the probe motivates it and
   does not stand in for it.

## 4. What exists today

### 4.1 The start rules and how a source is resolved

_Verified_ by reading `fit/starts.rs` and `fit/chain_starts.rs`. `ChainStarts`
is `Spread(Spread) | Point(Point)` (`starts.rs:47-54`). The sourced rules are
`Spread::FromPosterior { source }` — one row per chain drawn uniformly with
replacement from a stored posterior's `draws.tsv` (`chain_starts.rs:928-1013`) —
`Point::FromMle { source }` — every chain at the point a leaf's `fit_state.toml`
records in `start_values` (`:1021-1041`) — and `Point::FromParams { path }`. A
handle (`@label`, hash prefix, run directory, `fit.toml`) resolves to one method
leaf through `resolve_fit_leaf` (`chain_starts.rs:453-499`); a segment holding
several leaves is refused with the leaves listed. `resolve_starts` (`:407-445`)
runs before the model is compiled (`fit/mod.rs:383-397`) and folds the source
file's content digest into the method level's `deps` (`fit/mod.rs:1049-1050`),
so rewriting a source in place re-keys the run (gh#541).

Before any chain runs, `check_source_converged` (`fit/gating.rs:439-488`)
refuses a source whose stored verdict is not converged, unless
`--allow-nonconverged-source` is passed (`args/mod.rs:683-687`).

### 4.2 The IF2 check this rule reuses

_Verified_, `fit/gating.rs:182-279`. An IF2 leaf is judged by two legs. The
first is Â, a per-parameter chain-agreement statistic on the last half of the
iterations, against `a_thresh` (default 1.01). The second is the spread of the
chains' clean-eval log-likelihoods:

```rust
// fit/gating.rs:250-257
let delta_db = (hi - lo) * NATS_TO_DB;
let sigma_max = scout.chain_eval_ses.iter().cloned().fold(0.0_f64, f64::max);
let se_floor_db = 8.0 * sigma_max * NATS_TO_DB;
let threshold_db = gate.decibans_thresh.max(se_floor_db);
```

and the fit is refused (`DecibansSpread`) when `delta_db >= threshold_db`: the
best and worst chains are further apart than both the configured threshold and
eight Monte Carlo standard errors. The refusal message says "its chains landed
in different basins" (`gating.rs:298`). For this rule that verdict is the
expected input, not a defect (§7.9).

### 4.3 An IF2 leaf already stores every chain's end point

_Verified_ by reading the writer and a stored leaf. `write_clean_eval_tsv`
(`fit/runner.rs:3441-3466`), called for every IF2 leaf (`fit/mod.rs:1412`),
writes `chain_evaluations.tsv`: one row per chain that reached the clean
evaluation, with columns
`chain, loglik, se, ess_mean, ess_min, ess_min_step,
n_neg_inf_incr` and then
one column per estimated parameter. The parameter values are written with `{}`
formatting, which round-trips an `f64` (the gh#266 comment at `runner.rs:3462`).
A stored row, from
`.../multistart/explicit_59/if2_07/out/fits/fit-0432ee93/if2-763dc04d/seed_1-06cbd6b3/chain_evaluations.tsv`
on `feat/bsplines-s2`:

```
chain	loglik	se	ess_mean	ess_min	ess_min_step	n_neg_inf_incr	b1	b2	b3	b4	b5
0	-510.131886	0.034913	3790.79	1652.51	24	0	-0.6297136337778173	-1.3933625161221703	...
```

Chains IF2 dropped as filter-degenerate before the evaluation have no row
(`fit/state.rs:87-97`). The same leaf's `fit_state.toml` stores the threshold
the run was judged by (`resolved_gate`, `state.rs:125`), the start rule and its
kind (`chain_init_source`, `chain_starts_kind`, `:146`, `:154`), and
`start_values` — the winning chain's full parameter vector, including every
parameter the IF2 fit held fixed at its fixed value (`fit/mod.rs:1430`,
`collect_all_params`).

**So the data this rule needs already exists, and no IF2 output changes.** The
per-chain θ, ℓ_c and se_c are in `chain_evaluations.tsv`; the threshold, the
source's start kind and the source's fixed values are in `fit_state.toml`.

The `nl-*` optimisers write a different file, `chain_results.tsv`, with
`chain, loglik, status, n_evals` and parameters formatted to 10 decimals
(`fit/nlopt_stage.rs:664-693`); they record no standard error, and they run on
the deterministic ODE backend (`fit/methods.rs:9`). §12 D1 explains why they are
not accepted.

### 4.4 A PGAS chain already draws its own latent path at its start

_Verified_, `rust/crates/sim/src/inference/pgas.rs:3850-3880`. On a fresh start
each chain obtains its initial reference trajectory from an unconditional
particle filter run at its own starting θ, on its own seed
(`initial_reference_trajectory`, `:3869`; the chain seed is
`derive_chain_seed(seed, chain_id)`, `cli/src/fit/pgas.rs:643`), following
Andrieu, Doucet and Holenstein (2010, _JRSS-B_ 72:269–342, §4.5). PMMH has no
latent path.

### 4.5 A start on a bound

On main today an IF2 end point can sit exactly on a bound: the log transform
clamps its inverse to the box (`rust/crates/sim/src/inference/types.rs:22`,
`:108`). The support-keyed transforms proposal as accepted on main decides to
nudge such a start inward (D3, `2026-10-29-support-keyed-transforms.md:1066`).
Its revision on `feat/support-transforms` (`f419b702`, not on main) reverses
that: a start given by the user, including a sourced rule, is refused with an
interior value to use, and a start the fit draws is redrawn, because a nudged
start lands at z ≈ ±23 and walks back through the tail for longer than burn-in
(PMMH on `sir`, bulk-ESS 14–17 against 444). The predicate is
`Transform::on_an_end` (`types.rs:186-189` on that branch). §7.7 follows the
revision.

## 5. Why this is not `init_mle`

`init_mle` started every sampler chain at the single point the IF2 stage
selected. The workflow-first proposal measured four chains starting at
`0.25424765281567646`, identical to 17 significant figures, while the record
called them four independent draws (§1.3 of that proposal). Three differences
make `from_optima` a different object, each enforced in code rather than in
documentation:

1. **N points, not one.** Each sampler chain gets a different IF2 chain's end
   point; no end point is used twice (D5). A start-collapse to one vector cannot
   happen by construction.
2. **The points come from dispersed searches, and the rule checks that.** The
   source must itself have been spread-started (`chain_starts_kind = "spread"`
   in its `fit_state.toml`); a point-started or unrecorded source is refused
   (D8). So the end points are the results of independent searches begun from
   dispersed draws, and when the likelihood has several optima the starts carry
   them.
3. **The record says what kind of start it was.** The kind is a third value,
   `optima`, not `spread`, and `fit summary` says in words that R̂ compares
   chains begun at IF2 optima (§7.10). The `init_mle` failure was as much the
   record's claim of independence as the collapse.

The honest residual: when the likelihood has one optimum, the IF2 end points
cluster within IF2's own Monte Carlo noise, and the chains begin closer together
than prior draws would put them. R̂ is then weaker evidence of exploration than
under `from_prior` — not void, as under a point start, but weaker. The summary
measures that per parameter (the start-spread ratio of §7.10) and says so when
the starts were narrower than the posterior. §9 measures whether this weakness
produces false agreement in practice.

## 6. The literature supports the move by analogy, and the analogy is weaker in two places

Gelman, Vehtari, McElreath et al. (2026), _Bayesian Workflow_ (the PDF in the
repository root, page numbers as printed), recommend the same shape of procedure
for multimodal or hard posteriors:

> "Instead of running a large number of costly Markov chains, we can run the
> much faster Pathfinder algorithm (see Section 13.2) from different initial
> values to help find more modes, and then start fewer Markov chains based on
> the found modes. Pathfinder can also discard modes that have negligible
> posterior mass so that the further computation is focused where it matters."
> (§12.3, pp. 217–218)

and, from the case study where it is applied, "the Pathfinder draws can be used
to initialize Stan's HMC as they are likely to be closer to where most of the
posterior mass is than the default Stan initialization" (§27.4, p. 426).
Pathfinder is Zhang, Carpenter, Gelman and Vehtari (2022), "Pathfinder: Parallel
quasi-Newton variational inference", _JMLR_ 23(306).

`from_optima` substitutes IF2 for Pathfinder in that procedure. The substitution
is an analogy, and it is weaker in two respects the measurement must answer:

- **IF2 maximises the likelihood; Pathfinder approximates the posterior.** IF2
  ignores the prior (Ionides et al. 2015), so its end points are likelihood
  optima, and Pathfinder's L-BFGS paths climb the log posterior. Where the prior
  is informative relative to the data, a likelihood optimum can sit where the
  prior puts little mass. §7.7 refuses candidates where the log prior is not
  finite; for the rest, the sampler's warm-up moves the chain, and §9 includes a
  fixture with informative priors (the `smooth` spline form).
- **IF2 returns points; Pathfinder returns approximate draws with weights.**
  Multi-Pathfinder importance-resamples from a mixture of normal approximations
  (§13.2, p. 242), so its starts are dispersed like the posterior and its modes
  are weighted by approximate mass. IF2 end points carry only a log-likelihood.
  The good-chain test (§7.4) is a crude mass filter, and §7.5 deliberately
  spreads the starts across the distinct optima found rather than weighting by
  mass, because the purpose of starts is diagnostic: "we want dispersed initial
  points in order to have reliable convergence diagnostics and to potentially
  explore all the relevant modes" (§30.5, p. 469). The book also warns that "the
  probability of chains ending in different modes can be different from the
  relative probability mass of each mode, and running more chains doesn't fix
  this" (§12.3, p. 218); `from_optima` does not fix that either, and does not
  claim to.

The book also warns against the step the good-chain test automates. Only after
understanding the posterior geometry "—and only then!—we can discard Markov
chains stuck in minor modes or refine our initialization scheme" (§12.5, p.
234), and "we do not advocate mindlessly discarding misbehaving chains" (§30.5,
p. 469). The test discards IF2 optima, before any sampler runs, by a threshold.
Three things keep it from being mindless: the threshold is the one the user set
for, and saw applied in, the source's own `fit summary`; every discarded optimum
is listed with its log-likelihood in `optima_selection.tsv` and named in the
target's summary (§7.10); and the discards the probe actually needed are not
judgment calls — a gap of 1,373 nats is a likelihood ratio of about 10⁵⁹⁶. For
gaps near the threshold the record is the safeguard, and the summary's pointer
says how to inspect a discarded optimum.

**Why not `from_posterior` from a short run, which already exists?** The
workflow-first proposal named that as "the book's form of the same move" (§3.1,
its ruling 13). It does not address trapping: a short PGAS run from prior draws
traps chains in the same way, and `from_posterior` draws rows uniformly with
replacement from all chains' draws (`chain_starts.rs:1000-1002`), so a trapped
chain's draws are about a quarter of the pool and are drawn as starts in
proportion. IF2's perturbed search escaped the region the PGAS chain could not.
§9 includes this alternative as an arm, so the claim is measured rather than
asserted.

## 7. Design

### 7.1 Types

`fit/starts.rs` gains a third top-level variant and a third kind:

```rust
/// Where a method's chains begin.
pub enum ChainStarts {
    /// One independent draw per chain.
    Spread(Spread),
    /// Every chain at one vector. R̂ is not assessable; `fit summary` says so.
    Point(Point),
    /// Each chain at the end point of a different good chain of a stored IF2
    /// fit whose own chains were spread-started. R̂ compares chains begun at
    /// likelihood optima found from dispersed draws; `fit summary` says that,
    /// and how narrow the starts were relative to the posterior.
    Optima { source: Handle },
}

#[serde(rename_all = "snake_case")]
pub enum ChainStartsKind { Spread, Point, Optima }   // "optima" in fit_state.toml and run.json
```

`Spread::FromOptima` was the alternative. The module's own design statement is
that the kind is "the fact a reader of `fit summary` needs … The type makes that
fact a variant rather than a property one has to know per rule"
(`starts.rs:3-10`). What a reader of R̂ needs to know under this rule is neither
"independent draws" nor "one point", so it is a third variant; every `match` on
`ChainStarts` and `ChainStartsKind` then has to decide its case, which is the
point. A one-variant sub-enum (`Optima(Optima)`) is not introduced until a
second optima-sourced rule exists.

Method by method on `ChainStarts`: `kind()` → `Optima`; `is_point()` → `false`;
`tag()` → `"from_optima"`; `source()` → the handle; `describe()` →
`"from_optima @scout (each chain at its own IF2 optimum)"`;
`ignores_base_point(_)` → `true`; `from_parts` accepts `from_optima` and
`from-optima` as a sourced rule; `SOURCED_RULES` lists it. `can_redraw`
(`chain_starts.rs:165`) returns `false` (§7.8).

`fit/chain_starts.rs`:

```rust
pub enum InitSource {
    // … existing variants …
    /// The end point of source chain `source_chain`, as the source leaf's
    /// chain_evaluations.tsv records it.
    Optimum { source_chain: ChainIndex, path: PathBuf },
}

/// One source chain's end point.
pub struct Optimum {
    pub chain: ChainIndex,
    pub loglik: f64,
    pub se: f64,
    /// Estimated-parameter values in the source, natural scale.
    pub theta: BTreeMap<String, f64>,
}

/// What the selection decided for one source chain. One row of
/// optima_selection.tsv.
pub enum OptimumStatus {
    /// Used: the start of target chain `target_chain`.
    Selected { target_chain: ChainIndex },
    /// Good, but the target had fewer chains than good candidates (§7.5).
    GoodUnused,
    /// Failed the good-chain test (§7.4).
    BelowThreshold { delta_db: f64, threshold_db: f64 },
    /// Good, but a parameter sits on a finite end of its domain (§7.7).
    OnBound { param: String, bound: f64 },
    /// Good, but the target's log prior is not finite there (§7.7).
    ZeroPrior { param: String },
}

/// The good-chain test alone, shared by the target's selection and the IF2
/// leaf's `fit summary` so the two cannot disagree (§7.4).
pub fn is_good(best: &Optimum, c: &Optimum, decibans_thresh: f64) -> bool;

/// Read the source, classify every chain, choose n_chains of the good ones.
pub fn select_optima(
    ctx: &StartContext<'_>,
    source: &OptimaSource,
    n_chains: usize,
) -> Result<OptimaSelection, InitError>;

pub struct OptimaSource {
    pub leaf: PathBuf,
    pub optima: Vec<Optimum>,          // chain_evaluations.tsv
    pub decibans_thresh: f64,          // fit_state.toml resolved_gate, else GateConfig::default()
    pub held_fixed: BTreeMap<String, f64>, // source fit_state.toml start_values minus its estimated set
}

pub struct OptimaSelection {
    pub rows: Vec<(Optimum, OptimumStatus)>,
    pub starts: Vec<ChainStart>,       // n_chains, InitSource::Optimum each
    /// Target-estimated parameters the source held fixed; every chain starts
    /// at the source's fixed value (§7.6).
    pub common_start_params: Vec<String>,
}
```

`ResolvedSource` (`chain_starts.rs:352-362`) changes `dep: ArtifactRef` to
`deps: Vec<ArtifactRef>`. The existing rules produce a one-element vector, so
the `deps` the method level folds (`fit/mod.rs:1049-1050`, already a `Vec`) are
the same list as today and their method hashes do not move. That last claim is
_inferred_; the implementation's first commit tests it (§10, step 1).

New `InitError` variants, each with a message that names the remedy:

```rust
pub enum InitError {
    // … existing variants …
    /// Fewer good, usable source chains than target chains.
    TooFewOptima {
        needed: usize, usable: usize, evaluated: usize, source_chains: usize,
        excluded: Vec<(ChainIndex, OptimumStatus)>,
    },
    /// The source leaf is not an IF2 leaf.
    OptimaSourceNotIf2 { leaf: PathBuf, method: String },
    /// The source's own chains began at one point, or its start kind was not
    /// recorded.
    OptimaSourceNotSpread { leaf: PathBuf, rule: Option<String> },
}
```

Two refusals are configuration errors, raised where the method is validated
rather than at the draw: `from_optima` on a method other than `pgas` or `pmmh`
(D2), and `--allow-nonconverged-source` together with `from_optima` (D11).

### 7.2 Wire form and parsing

```toml
starts = { from_optima = "@scout" } # a fit handle; resolved to one IF2 leaf
```

`--starts from_optima=@scout` on the command line. The source is always a fit
handle; unlike `from_posterior`, no file path form is accepted, because the rule
reads two files of one leaf together (§7.3). Serialization is the one-key map
`{"from_optima": "@scout"}`, the shape `Serialize for ChainStarts` already
produces for sourced rules (`starts.rs:329-345`).

### 7.3 Resolution

`resolve_starts` gains an `Optima` arm, run at the same point as today's sourced
rules (before compile):

1. `resolve_fit_leaf(handle)` (unchanged).
2. The leaf's `fit_state.toml` must say `method = "if2"`, else
   `OptimaSourceNotIf2`.
3. Its `chain_starts_kind` must be `spread`, else `OptimaSourceNotSpread` (D8).
4. `chain_evaluations.tsv` must exist and carry a column for every parameter the
   target estimates or a fixed value for it in `start_values` (§7.6), else the
   existing `SchemaMismatch`, which `from_posterior` already uses for a missing
   column (gh#274).
5. `check_source_converged` is not called (§7.9).
6. `deps` = the leaf digests of `chain_evaluations.tsv` and `fit_state.toml`,
   through the existing `cas_leaf_file_dep` (`fit/cas.rs:658`).

The selection itself needs the target's transforms and priors, so it runs at the
draw (`draw_chain_starts`), from the resolved files.

### 7.4 Which chains are good: the IF2 spread check applied to a pair

Let b be the source chain with the highest ℓ. Chain c is good when the IF2
chain-spread check of §4.2, applied to the two chains {b, c}, would not refuse:

```
(ℓ_b − ℓ_c) · NATS_TO_DB  <  max(decibans_thresh, 8 · max(se_b, se_c) · NATS_TO_DB)
```

with `decibans_thresh` the value in the source's `resolved_gate`, or
`GateConfig::default()` (30.0) when a legacy `fit_state.toml` has none — the
same fallback `check_source_converged` uses (`gating.rs:443`). Chain b is always
good. A row with a non-finite ℓ or a non-zero `n_neg_inf_incr` (the filter lost
the data at some step) is not good.

The check keeps the IF2 check's two terms and their meaning — a configured
evidence threshold and a Monte Carlo noise floor eight standard errors wide —
and applies them to the comparison this rule makes, each chain against the best,
instead of to the best-against-worst spread. Using the whole run's `max(se)`
instead would let one noisy trapped chain widen every other chain's tolerance; a
trapped chain's se is the largest in the probe (5.17 nats against at most 0.19,
§3). The pairwise form lets a noisy chain widen only its own tolerance. In the
probe's trapped case that tolerance is 8 × 5.17 = 41 nats, against a gap of
1,373; it is excluded either way.

The threshold is the source's, not the target's, so the set of good chains is a
fact about the stored IF2 fit, the same set the IF2 leaf's `fit summary` shows
(§7.10). Changing it means re-running IF2 with `--decibans-thresh`; the target
has no override (D3).

### 7.5 Which good chains are used, and what happens when there are too few

The usable candidates are the good chains minus those §7.7 excludes. With n
target chains and u usable candidates:

- **u < n: refuse** with `TooFewOptima`. No end point is reused and none is
  jittered. The message lists every excluded chain with its reason and
  recommends a source chain count: `ceil(n · source_chains / usable)`, rounded
  up to a multiple of 4. For example, with 16 IF2 chains of which 3 are usable
  and 4 target chains: "re-run `@scout` with `chains = 24`". The recommendation
  is arithmetic on the observed yield, and the message says so.
- **u = n:** all are used.
- **u > n: choose n far apart.** Map each candidate to the target's
  unconstrained coordinates z (each estimated parameter through its target
  transform), standardise each coordinate by its standard deviation across the
  usable candidates (dropping coordinates with zero spread), and select by
  farthest-point: start with b, then repeatedly add the candidate whose
  Euclidean distance to the nearest already-selected candidate is largest; ties
  go to higher ℓ, then lower chain number. Target chain k gets the k-th selected
  candidate, so chain 0 is always the best optimum.

"Distinct" therefore means distinct IF2 _chains_ — independent searches — and
not distinct _basins_. Requiring n geometrically separate optima would refuse
every model whose likelihood has one optimum, which is the common case and was
every dataset in the probe (§3, qualification 3). The farthest-point choice
applies only when there is a surplus, and it spends the surplus on spreading the
starts across whatever separate optima IF2 found, which is what the starts are
for (§6). The selection is a deterministic function of the source files and the
target's transforms; it draws no random numbers.

### 7.6 Parameters whose role differs between source and target

The partition into estimated and fixed parameters can differ between the two
files. Each case:

- **Estimated by both:** the start is the source chain's value.
- **Fixed in the target, estimated in the source:** the target's fixed value
  wins. That is the existing seam: a `ChainStart` holds only the target's
  estimated parameters (`chain_starts.rs:12-17`). A notice names the parameters,
  since the optima were found with them free.
- **Estimated in the target, fixed in the source:** every chain starts at the
  source's fixed value, read from the source's `start_values`. This is the
  `smooth` spline case (τ, §3) and any hierarchical scale IF2 cannot maximise.
  Because all chains share that coordinate's start, that parameter's R̂ is
  reported as not assessed with the reason "every chain started at the value the
  source held fixed", a new `RhatRefusal::CommonStart` beside the existing
  `PointStart` (`sim/src/inference/convergence.rs:136`), with
  `is_pathology() = false`. The other parameters' R̂ are unaffected.
- **Absent from the source altogether** (neither estimated nor fixed):
  `SchemaMismatch`, as `from_posterior` does.

The source's model identity and data hashes (its segment's `fit.meta.json`) are
compared with the target's once the target's are computed, before any chain
runs. A difference prints a notice and is written into `optima_selection.tsv`'s
header; it is not refused, because a start never changes the target density (§1)
and scouting on a variant model or a data subset is a legitimate use (D12).

### 7.7 Starts on a bound, and starts the prior rules out

A usable candidate must have every target-estimated parameter strictly inside
its effective domain (`Transform::on_an_end` is false) and a finite target log
prior. A good candidate failing either is excluded with status `OnBound` or
`ZeroPrior` and counts against u in §7.5.

Exclusion is the drawn-start branch of the support-keyed revision's D3 (§4.5),
not the given-start branch. The IF2 end point is the output of a stochastic
search, not a value the user wrote, and the rule has other candidates to take
instead. Refusing the whole run for one on-bound candidate would throw those
away, and nudging it inward would put a chain at z ≈ ±23, the tail the revision
measured as costing most of the ESS. When every good candidate is on a bound,
the refusal says that the likelihood optimum appears to lie on the bound and
names a profile or `nl-sbplx` as the test, as D3's message does.

### 7.8 Retry and refusal of a start the target cannot score

`can_redraw` returns `false`, so the gh#887 retry (`runner.rs:826-837`) does not
run, as for a point rule: there is no fresh draw to make, and substituting the
next candidate would make chain k's start depend on whether another chain's
start scored. A start the target's filter cannot score is refused by the
existing chain-start refusal of PGAS and PMMH, unchanged. Every usable candidate
scored a finite ℓ at the source's clean-evaluation particle count, so this
arises only when the target's filter has far fewer particles; the refusal text
for `from_optima` adds that comparison.

### 7.9 `--allow-nonconverged-source` is refused with this rule

The flag lifts `check_source_converged`'s refusal of a source whose chains
disagree. Under `from_optima` chain disagreement is the input — the rule exists
to use the separate optima the chains found — so the check is not run (§7.3,
step 5) and the flag has nothing to lift. Accepting it silently would invite the
reading that it relaxes the good-chain test. `fit run` refuses the combination:
"`--allow-nonconverged-source` has no effect with `from_optima`, which does not
require the source's chains to agree; it uses the chains within the source's
decibans threshold of the best. To admit more chains, run more IF2 chains, or
re-run IF2 with a larger `--decibans-thresh`."

At resolution the target prints one line naming the source's own verdict, so the
user sees it was read: "source @scout: IF2 chains did not agree (Â 1.31 on b3;
spread 1,372.8 nats). Expected input for from_optima: 11 of 16 chains are within
30.0 dB of the best."

### 7.10 The record, and what `fit summary` says R̂ means

**`chain_starts.tsv`** (the one writer, `chain_starts.rs:1289`): the `source`
column is `from_optima:source-chain-<j>`, so each target chain's row names the
IF2 chain it came from. The header line adds the source leaf's `run_id`.

**`optima_selection.tsv`** (new, at the target leaf root): one row per source
chain listed in `chain_evaluations.tsv`, with columns
`source_chain, loglik, se,
delta_db, threshold_db, status, target_chain`, where
`status` is one of `selected` (used; `target_chain` says by which chain),
`good_unused` (good, but not needed), `below_threshold` (failed §7.4),
`on_bound` and `zero_prior` (§7.7), and `delta_db` is the gap to the best chain.
A header comment records the source handle and leaf, the threshold and where it
came from, the parameters held fixed in the source, and any model or data
mismatch. This file is what makes a discard inspectable rather than silent (§6).

**`fit_state.toml` and `run.json`**: `chain_starts_kind = "optima"` and
`chain_init_source = "from_optima @scout"`, through the existing fields. No new
`fit_state.toml` field; the summary reads `optima_selection.tsv`.

**`fit summary` of the target**, under the identity line:

```
seeded from:  from_optima @scout (each chain at its own IF2 optimum: 4 used of 11 good of 16 IF2 chains)
R̂ here compares chains started at the end points of independent IF2 searches begun from
prior draws. It tests whether the chains agree after warm-up; it is not a test from
over-dispersed starts.
start spread / posterior sd:  b3 0.42, b4 0.61  (below 1: the starts were narrower than
the posterior, so R̂ near 1 is weaker evidence there than under from_prior)
not used:  5 IF2 optima more than 30.0 dB below the best (chains 2, 5, 9, 12, 14);
see optima_selection.tsv, and score one with `camdl pfilter --params <leaf>/chain_<j>/final_params.toml`
```

The R̂ band and the verdict thresholds are those of every sampler fit; only the
explanation is added. The start-spread ratio is, per estimated parameter, the
standard deviation of the chains' starting values divided by the posterior
standard deviation from `draws.tsv`, both on the natural scale; the line lists
the parameters below 1, or says "none" when there are none. It is computed at
summary time from files the leaf already holds. Parameters under §7.6's common
start show
`R̂ — not assessed: every chain started at the value the source held
fixed`.

**`fit summary` of the IF2 source** gains one line, from the same `is_good`:
"for from_optima: 11 of 16 chains within 30.0 dB of the best (chains …)". That
is step 3 of the workflow: the user sees how many good chains there are before
running the sampler.

### 7.11 Run identity

The rule enters the method level's identity as `{"from_optima": "@scout"}`, as
every sourced rule does (`Serialize for ChainStarts`), and the two source
digests enter its `deps` (§7.3). Everything the selection reads is therefore
keyed: the end points, logliks and ses (`chain_evaluations.tsv`), the threshold,
the source's start kind and fixed values (`fit_state.toml`), and, from the
target's own identity, its chain count, transforms and priors. Re-running the
source under the same label produces a new leaf with new digests and re-keys the
target. The selection draws no random numbers, so the seed does not enter it;
the seed still keys the sampler. No `ir/VERSION` change, no golden change.

## 8. Recommended use

The IF2 source should have about four times as many chains as the sampler: 16
IF2 chains for 4 PGAS chains. The probe does not settle the number. It ran 5 to
8 single-chain IF2 fits per dataset and records the fits run and the distinct
good solutions (`multistart_summary.tsv`), not how many fits passed its 5-nat
test, so the yield of good chains per IF2 chain is not yet measured. A 4×
surplus is chosen to leave room for a trapped or on-bound minority of a quarter
or more. It is a recommendation for `docs/` and the refusal message, not a rule;
§9 measures the yield per fixture (M4) and the note revises the number if it is
wrong.

## 9. The measurement that must precede merge

Per `CLAUDE.md` and `docs/dev/agent-verification-conventions.md` ("A new
inference method merges with the note that measures it"), the branch merges only
with a `docs/dev/notes/` note that runs the comparison below on committed
fixtures, cited from the merge commit. The criteria are fixed here, before any
run.

### 9.1 Fixtures

1. **`spline_seir`**, both forms, data seeds 23, 31, 47, 59
   (`tests/fixtures/bspline_inference/` and `tests/recovery/cases/spline_seir/`
   from `feat/bsplines-s2`). The fixture of §3, with the `smooth` form
   exercising §7.6 (τ held fixed in IF2) and informative priors (§6). PGAS, 4
   chains, 64 particles, 1,000 adaptation and 3,000 kept sweeps, as in §3.
2. **`seir_age`** (`tests/recovery/cases/seir_age/`, on main, with committed
   `if2.toml`, `pgas.toml` and `pmmh.toml`). The non-spline case, run under both
   PGAS and PMMH with the committed configurations.
3. **An Ebola-like province fixture**, new:
   `tests/recovery/cases/ebola_province/`, a reduced copy of the two-province
   model that
   `docs/dev/notes/2026-08-24-pgas-binomial-sampler-is-half-the-fit.md` profiles
   (`ebola-bdbv-camdl/models/bvd_province_nu_carec.camdl`; two provinces, the
   exposed stage in three sub-stages), vendored with its source commit recorded
   in the case README, and synthetic data simulated from a committed
   `truth.toml` at province populations. It is the regime gh#876 measured, where
   prior-draw starts are often starts the filter cannot score, so it also tests
   the gh#887 retry counts.

### 9.2 Arms

Each arm per fixture × dataset is run at 5 run seeds:

- **A — `from_prior`**, today's default.
- **B — `from_optima`**: IF2 with 4× the sampler's chains,
  `--starts
  from_prior`, the fixture's `if2.toml` otherwise; then the sampler
  with `--starts from_optima=@<label>`.
- **C — `from_posterior` from a short run**: a `from_prior` PGAS run whose wall
  time matches B's IF2 run, then the sampler with
  `--starts
  from_posterior=@<label>`. The existing alternative of §6.

A reference posterior per dataset: arm A with 16 chains and 4× the kept sweeps,
with trapped chains (§9.3) removed and the removal stated.

### 9.3 Quantities

- **Trapped-chain count** per arm: a chain is trapped when the marginal
  log-likelihood at its last kept θ — `camdl pfilter`, 2,000 particles, five
  seeds, the `stuck_chains.py` protocol — is below the run's best chain's by
  more than §7.4's threshold. Reported as count of total chains, with a
  Clopper–Pearson 95% interval.
- **Agreement with the reference**: for each estimated parameter (and, on
  `spline_seir`, log β(t) at the 41 window times), z = (q_arm − q_ref) / MCSE at
  the 5%, 50% and 95% quantiles, MCSE (Monte Carlo standard error) from the
  arm's own draws; count |z| > 3. This is the check that concentrated starts
  have not produced agreement without exploration (§3, qualification 1).
- **R̂, bulk-ESS, tail-ESS** (worst over parameters), and divergent transitions.
- **Start-spread ratio** (§7.10) for arm B, per parameter.
- **Wall time**, total per arm including the IF2 or short-PGAS source run
  (`run.json` `inputs.wall_time_seconds`), and tail-ESS per total wall-second.
- **IF2 yield** for arm B: good / usable / total chains, and the number of
  refusals for too few optima.
- **Starts refused or retried** (gh#887) per arm, from `chain_starts.tsv`.

### 9.4 Merge criteria

The note states each criterion's outcome; the feature merges only if all hold.

- **M1 — trapping.** Arm A traps chains on at least two of the three fixtures
  (otherwise the rule is measured on one model only, and does not merge until a
  second committed fixture shows the problem). On each fixture where A traps, B
  traps fewer chains than A and no more than C.
- **M2 — no false agreement.** B's count of |z| > 3 is no greater than A's on
  A's untrapped runs, on every fixture.
- **M3 — cost where it is not needed.** On a fixture where A traps no chain, B's
  tail-ESS per total wall-second is at least half of A's.
- **M4 — refusals are rare at the recommended size.** With 4× IF2 chains, no
  more than one run in ten per fixture is refused for too few optima; if more,
  §8's recommendation is revised in the note before merge.

If M1 fails, the rule does not merge and the note records why. If only M3 or M4
fails, the note proposes the change (a different recommended size, or the rule
documented as for trapped cases only) and the maintainer decides before merge.

## 10. Implementation sequence

Each step is its own commit and green. The high-risk files of `CLAUDE.md` are
not edited: the selection lives in `fit/chain_starts.rs`, and PGAS and PMMH
receive starts through the existing `DrawnStarts` seam.

1. `ResolvedSource.dep` → `deps: Vec<ArtifactRef>`, with a test that the method
   hash of a `from_mle` and a `from_posterior` fit is byte-identical before and
   after.
2. `ChainStarts::Optima`, `ChainStartsKind::Optima`, parsing, serialization and
   the `fit_state.toml` round trip; every `match` updated; config validation of
   D2 and D11.
3. `resolve_starts`'s `Optima` arm (§7.3) with the three source refusals, each
   with a red-then-green test against a hand-built leaf.
4. `is_good`, `select_optima` (§7.4–7.7), `InitSource::Optimum`, and
   `optima_selection.tsv`. Tests: the pairwise threshold against a fixture of
   ses where the whole-run `max(se)` would admit a trapped chain; farthest-point
   order; the refusal text and its recommended count; `on_bound` and
   `zero_prior` exclusion; §7.6's three cases.
5. `RhatRefusal::CommonStart` and the per-parameter demotion.
6. `fit summary` lines for target and source (§7.10).
7. The measurement note (§9), on the branch, before merge.

**Sequencing.** The branch is cut after the support-keyed transforms series
lands on main, because §7.7 uses `Transform::on_an_end` and follows that series'
revised D3; and the measurement runs after `feat/bsplines-s2`'s `spline_seir`
fixture is on main, so the note cites committed fixtures on the merge target.

## 11. What changes for users

A new `--starts` value and `starts` key; a new start kind `optima` in
`fit_state.toml`, `run.json` and `chain_starts.tsv`; a new sidecar
`optima_selection.tsv`; two new `fit summary` passages. Nothing existing changes
behaviour or identity (step 1's test). `--allow-nonconverged-source` is refused
in one new combination.

## 12. Decisions

**D1 — Which source fits are accepted? IF2 only. Leaning.** The `nl-*`
optimisers maximise the deterministic ODE likelihood (`fit/methods.rs:9`), a
different likelihood from the stochastic one PGAS and PMMH sample, so their
optima are not optima of the target's problem even on the same model. They
record no standard error, so §7.4's noise floor has no input, and their status
column would need its own rules. The natural pairing for them is `nl-*` → `mh`
(the ODE sampler), which needs its own measurement on an ODE fixture; until a
proposal brings one, an `nl-*` source is refused with a message naming
`from_mle` and `from_prior`. Including `nl-*` now would admit the pairing with
no evidence; excluding it costs nothing anyone uses today.

**D2 — Which samplers accept the rule? PGAS and PMMH. Solid.** Both sample the
chain-binomial likelihood IF2 maximises, and both take starts through the same
`DrawnStarts` seam. `mh` is refused (ODE, D1). IF2 and the optimisers are
refused: an IF2 run from IF2 optima is a restart of the same search, and an
optimiser has no between-chain diagnostic for the starts to serve.

**D3 — Which chains are good, and whose threshold? The IF2 spread check applied
to each chain paired with the best, at the source's `decibans_thresh`.
Leaning.** The alternative, the check's own whole-run `max(se)`, lets the
noisiest chain — usually the trapped one — widen everyone's tolerance (§7.4).
Using the source's threshold makes the good set a stored fact the source's
summary already shows; the cost is that changing it means re-running IF2. A
target-side override was rejected because it would make the same IF2 leaf mean
different things to different targets, and the IF2 summary could no longer say
which chains a target will use. Leaning rather than solid only because the
pairwise form is new and its behaviour on near-threshold chains is first seen in
§9.

**D4 — What makes optima "distinct", and which are used when there are more than
needed? Distinct IF2 chains; farthest-point in standardised unconstrained
coordinates. Leaning.** Geometric distinctness would refuse every unimodal
likelihood (§7.5). Taking the top n by ℓ was the alternative; it concentrates
the starts in the best basin, which is what weakens R̂ (§5). Farthest-point
spreads them across the optima found, which is the diagnostic purpose of starts
(§6). It does not weight by mass, and the summary does not claim it does.

**D5 — Too few good chains: refuse, with the IF2 chain count to run. Solid.**
Reusing an end point, with or without jitter, puts two chains at one point (or
at one point plus a noise scale nobody chose), which is the collapse §5 exists
to prevent, made smaller. Refusing costs one IF2 re-run, and the message says
how many chains.

**D6 — Candidates on a bound or outside the prior's support: excluded and
counted, not nudged, not refused. Solid.** §7.7. Follows the support-keyed
revision's drawn-start branch for the reason that revision measured.

**D7 — Latent path for PGAS: unchanged, one unconditional particle filter pass
per chain at its own start and seed. Solid.** §4.4. IF2 stores no latent path,
and its swarm's states were produced at perturbed parameters, so there is
nothing better to import. Each chain's path is already its own.

**D8 — The source must have been spread-started. Solid.** This is the check that
makes "optima found from dispersed draws" true rather than hoped (§5). A
point-started IF2 source would give end points whose spread reflects only IF2's
noise around one search; legacy leaves with no recorded kind are refused with
"re-run the source".

**D9 — Run identity: the rule plus the digests of `chain_evaluations.tsv` and
`fit_state.toml`; `ResolvedSource.deps` becomes a vector. Solid.** §7.11. Every
input to the selection is keyed; existing rules' hashes are tested unchanged.

**D10 — Record and summary: a third start kind `optima`; R̂ reported with its
meaning stated; a start-spread ratio; every discarded optimum listed. Leaning on
the ratio's cut of 1.** The kind is the design's main safeguard against
`init_mle`'s second failure, a record that misdescribed the starts. The ratio's
cut at 1 (starts narrower than the posterior) is a plain reading of
"over-dispersed" relative to the target, not a calibrated threshold; §9 reports
the ratios observed, and the cut can move in the note before merge.

**D11 — `--allow-nonconverged-source` with `from_optima`: refused. Leaning.**
The flag would be a no-op (§7.9); a silent no-op on a flag whose name suggests
relaxing a check is the kind of thing a user acts on wrongly. The alternative,
accepting it with a warning, keeps scripts that pass it everywhere working; in
alpha that is not a constraint.

**D12 — Role and problem mismatches between source and target: source-fixed
parameters start at the source's value with their R̂ not assessed; target-fixed
parameters keep the target's value; a model or data mismatch is a notice.
Leaning.** §7.6. Refusing source-fixed parameters (the `from_posterior`
precedent) would make the rule unusable for any parameter IF2 cannot maximise,
including the case that motivated it (τ in the `smooth` form, §3). Drawing them
from the prior instead would pair an optimum conditional on one τ with a
different τ, which is not an optimum of anything, and in a non-centred form like
`smooth` changes the curve the other coordinates describe. Taking the source's
value and withholding only that parameter's R̂ is what the start actually is,
said accurately. A mismatch of model or data is a notice because a start cannot
change the estimand (§1).

**D13 — Merge only with §9's note, under M1–M4. Solid.** The probe is one
fixture, its R̂ improvement is partly what concentrated starts do by
construction, and its procedure differs from this rule in five ways (§3).

**D14 — Sequencing: after the support-keyed transforms series and the
`spline_seir` fixture are on main. Solid.** §10.

## 13. Out of scope

- **`nl-*` sources and the `mh` target** (D1): a separate proposal with an ODE
  fixture measurement.
- **Weighting chains by mode mass** (stacking, Yao, Vehtari and Gelman 2022,
  cited by the book at §12.3, p. 218): a posterior-combination method, not a
  start rule.
- **Laplace- or Pathfinder-style draws around each optimum**: needs a curvature
  estimate IF2 does not produce; it would also be a new inference method with
  its own measurement.
- **Changing what IF2 writes**: not needed (§4.3).
