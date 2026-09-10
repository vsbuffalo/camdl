# Workflow-first fit config: a method, not stages; checks as fit verbs

**Status:** Proposal **Date:** 2026-09-08 **Motivation:** the `[stages]` half of
`fit.toml` is a pipeline engine whose one composition primitive, `init_mle`,
destroys the property that convergence diagnostics need, and whose vocabulary
(an algorithm per entry) cannot express the workflow steps the Bayesian workflow
literature treats as core. Replace it with a problem/method split, one method
per file, and workflow steps as `fit` subcommands.

**Scope:** `rust/crates/cli/src/fit/config_v2.rs` (`FitConfigV2`, `Stage`,
`StartsFrom`), `fit/init.rs` (`InitMethod`), the `fit run` stage loop in
`fit/mod.rs`, `fit/gating.rs`, the four non-fit readers of the config
(`simulate --draws prior --fit`, `pfilter --fit`, `survey --fit`,
`profile --fit`), and the fit-store path shape. **Audience:** camdl
contributors; the maintainer decisions are in §8.

Terms used throughout, defined once. _R̂_ (R-hat) is the potential scale
reduction factor: the between-chain spread of a quantity divided by its
within-chain spread, computed over several independent Markov chains; it is near
1 when the chains agree and grows when they do not. _ESS_ is the effective
sample size, the number of independent draws a correlated chain is worth. _MLE_
is a maximum-likelihood point estimate. _IF2_ is iterated filtering, a
particle-filter optimizer that returns an MLE; _PGAS_ (particle Gibbs with
ancestor sampling) and _PMMH_ (particle marginal Metropolis–Hastings) are the
two particle-filter posterior samplers; _NUTS_ and _MH_ are the ODE-backend
samplers. A _prior predictive_ is data simulated from the model with parameters
drawn from the priors, before any data are fitted; a _posterior predictive_ is
the same with parameters drawn from a fitted posterior. _SBC_ is
simulation-based calibration checking: draw parameters from the prior, simulate
data, fit, and check that the true parameter's rank among posterior draws is
uniform over many repetitions. A _handle_ is any of the fit references
`FitRef::classify` already accepts: `@label`, a fit-id hash prefix, a run
directory, or a `fit.toml` path (`fit/handle.rs:26`).

---

## 0. Summary

Split the file into the half every command reads and the half only `fit run`
reads. The first half is the _problem_: model, data, the estimate/fixed
partition, scenario, simulator settings. The second is one _method_
(`[method]`): one algorithm with its knobs and a typed chain-starts policy. A
file is one (problem, method) pair; a second way of fitting the same problem is
a second file. The two share the fit-level hash, which is computed from the
problem alone, and sit in sibling store segments labelled by their file stems —
which is what lets a per-file `--label` or `@handle` resolve to exactly one
leaf. The store already factors this way — the fit-level digest excludes
`[stages.*]` (`fit/cas.rs:5-8`) and the stage level is the method — so the
config is being brought into line with the identity model, not the other way
round.

In-file chaining (`init_mle = "<stage>"`) is removed. Warm-starting stays, as an
explicit `starts = { from_posterior = "@handle" }` (one posterior draw per
chain, which preserves the diagnostic) or `starts = { from_mle = "@handle" }`
(every chain at one point, the escape hatch). When a multi-chain sampler is
started from a point, `fit summary` reports R̂ as _not assessed_ with the reason,
never as a pass; optimizer-only runs are exempt because they report no R̂.

Workflow steps become verbs under `fit`, the namespace of everything that takes
a `fit.toml`: `camdl fit preflight` (the observation design, the filter budget,
the forecast horizon, and the prior predictive on the data's own windows —
everything a modeller should know before spending the compute; needs only the
problem), `camdl fit recovery` (fixed-truth parameter recovery over replicates,
the existing `[synthetic]` block with its missing roll-up), and a reserved
`camdl fit calibration` (SBC, named and shaped here, not built). Posterior
predictive checking stays `fit predict`. `camdl check` stays the static model
check it is today. Every check _computes_ automatically where it is cheap,
_fails_ only on deterministic facts about the model as written, _reports_
frequencies without a threshold, and _never passes_. `fit run` runs preflight's
deterministic half before any sampling, because a check that costs seconds and
is skipped by default is a check nobody runs.

Orchestration of several verbs stays outside camdl. The boundary is a test:
_does any step between the verbs need a judgement camdl cannot make?_ If not
(simulate → fit → compare-to-truth), the sequence is one verb. If so (is this
prior plausible; is this fit trustworthy), it is a script, and camdl's job is to
make each verb idempotent, memoized, and addressable by handle, which it already
is.

The change re-keys every stored fit (the fit-level digest changes because a dead
field leaves it; the method-level payload changes shape). It touches 67 test
files, 42 documentation files, and 6 committed fixture configs. A `[stages]`
block is rejected at load with the exact rewrite, in the pattern the
`starts_from` → `init_mle` rename already uses.

---

## 1. What exists today

Everything in this section is verified against `main` at `b44e5b71`; the command
that verified each claim is given inline.

### 1.1 The file and its types

`fit.toml` deserializes into `FitConfigV2` (`config_v2.rs:26-135`). Three fields
have no `#[serde(default)]` and are therefore structurally mandatory:
`estimate`, `fixed`, and `stages`. `stages` is `IndexMap<String, Stage>`; the
`IndexMap` is load-bearing because stages execute in declaration order.

`Stage` (`config_v2.rs:875`, `#[serde(tag = "algorithm")]`) has eight variants,
not five:

```
awk 'NR>=875 && NR<=1300' rust/crates/cli/src/fit/config_v2.rs \
  | grep -o 'rename = "[a-z0-9-]*"' | grep -v '"init"' | sort -u
→ if2, mh, nl-bobyqa, nl-sbplx, nuts, pfilter, pgas, pmmh
```

Every one is an inference algorithm or a likelihood evaluator. Each carries the
same three chain-start fields under different doc comments: `init`
(`InitMethod`, nine variants, `init.rs:69-130`), `init_mle` (`StartsFrom`, Rust
field `starts_from`, `config_v2.rs:2016`), and the `survey_path` /
`survey_top_k_n` companions that only `init = "survey_top_k"` reads. A fourth
chain-start surface, the top-level `fit_starts: Option<FitStarts>`
(`config_v2.rs:72`), is parsed, serialized into the fit-level identity hash,
consulted once to silence a warning (`config_v2.rs:2468`), and read by nothing
else:

```
rg -n 'fit_starts|FitStarts' rust/crates/cli/src --glob '!config_v2.rs'
→ no matches
```

Chaining is validated as a dependency graph (`validate_stage_dag`,
`config_v2.rs:2968`): a stage's `init_mle` must name an earlier stage or a
directory. At dispatch, a chained stage's declared `init` is ignored and every
chain is given the upstream point (`fit/mod.rs:1386-1476`; the recorded init
becomes `InitMethod::FromMle`, documented at `init.rs:118` as "all chains start
at the MLE point from a prior fit"). Two cross-stage gates ride on this: Gate 1
refuses a downstream stage whose upstream failed its chain-agreement check
unless `--allow-nonconverged-scout` is passed, and Gate 2 rejects a refine whose
best log-likelihood regressed below its scout's (`fit/gating.rs:1-18`,
`fit/mod.rs:1242,1485`).

### 1.2 What the store already knows

The fit store factors a fit into three identity levels — `fit · stage · seed`
(`docs/dev/cas-path-shape-contract.md`, kinds table). The fit level hashes the
canonical JSON of the config _minus_ `stages`, `fit_seeds`, and `output_dir`
(`fit_config_blob_hash`, `fit/cas.rs:422-427`). The stage level hashes one
stage's serialized fields minus its extension dimension, folded with its `deps`
(`Stage::identity_payload`, `config_v2.rs:1473`). The leaf address hashes only
the level hashes, not their names or labels (`runid/src/kind.rs:79-88`).

So the store's fit level _is_ the problem and its stage level _is_ the method.
The config's shape — one document, pipeline semantics, mandatory inference block
— is the thing out of step.

### 1.3 Four measured findings

**Chaining collapses every chain onto one point.** Same model, same seed, PGAS,
four chains. Chained off a priming stage via `init_mle`, all four chains start
at `0.25424765281567646`, identical to 17 significant figures. Unchained with
per-chain random init, the same seed gives `0.0934465282123066`,
`0.4447584453963407`, `0.07207300992324761`, `0.43115348618394506`. The priming
stage's own chains were properly spread and converged to that single point.

Why it matters is not a camdl detail. R̂ compares between-chain disagreement to
within-chain variance; chains that begin at one point begin in perfect
agreement, so an R̂ near 1 is guaranteed by construction and says nothing about
whether the posterior was explored. The workflow text is explicit that this is
what starts are for: "we want dispersed initial points in order to have reliable
convergence diagnostics and to potentially explore all the relevant modes"
(Gelman, Vehtari, McElreath et al. 2026, §30.5, p. 469), and its default
recommendation is "at least four independent chains" precisely because "multiple
chains are more likely to reveal multimodality and poor adaptation or mixing"
(§11.4, p. 198). A chained PGAS stage satisfies the letter of "four chains" and
none of the intent.

**The chain-origin record lied in four places; three are fixed.** The provenance
file reported four independent per-chain draws
(`uniform_unconstrained:chain-0 …`) beside four identical values (fixed
`21d06ee9`, gh#871); the IF2 stage's copy of that file was overwritten by a
second writer with no `source` column (fixed `69b1c270`, gh#872); the same value
was written as `from-mle` in one file and `from_mle` in two others (fixed
`e326b164`, gh#873). The fourth is open: `fit/state.rs:131-139` documents
`chain_init_source` as "surfaced as a one-line header in `camdl fit summary`",
and no such reader exists:

```
rg -n 'chain_init_source' rust/crates/cli/src/fit/fit_summary.rs
→ 3548:            chain_init_source: Some("lhs".into()),     (a test fixture; no read)
```

**A command that runs no inference requires an inference block.** A prior
predictive check — the one workflow step that by definition precedes fitting —
needs the model, its priors, the estimate/fixed partition, and the observation
design. Reproduced with the `tests/fixtures/mre` config minus its
`[stages.scout]` block:

```
camdl simulate model/sir_patches.camdl --draws prior -n 3 --fit fit_nostages.toml --output-dir results
error: error in fit_nostages.toml:
parse error: TOML parse error at line 1, column 1
missing field `stages`

  hint: `simulate --fit` expects a fit-config TOML, not a bare params file. It must declare a `[model]` table naming the model file, e.g.
    [model]
    camdl = "path.camdl"
```

The file declares `[model]`; the hint (`main.rs:3644-3655`, attached to every
load error on this path) points at the one table that is present. The same
`FitConfigV2::load` is the entry point for `pfilter --fit` (reads `[data]` only,
`pfilter.rs:878`), `survey --fit` (reads bounds, data, fixed, scenario,
`survey.rs:655`), and `profile --fit` (reads priors, bounds, fixed,
`profile.rs:344`). None reads `stages`; all require it.

**There is no workflow step type.** Nothing in the config or the CLI expresses
prior predictive checking as an artifact (`simulate --draws prior` writes raw
replicates to a user path, gh#711), fixed-truth recovery over replicates has no
roll-up (`[synthetic]` runs the fits; `coverage.tsv` is unbuilt, gh#154), and
SBC does not exist.

### 1.4 Who reads the config, and what they take

| reader                         | takes from the file                                            | reads `stages` |
| ------------------------------ | -------------------------------------------------------------- | -------------- |
| `fit run`                      | everything                                                     | yes            |
| `simulate --draws prior --fit` | `estimate` names, priors, `fixed`, `scenario`                  | no             |
| `simulate --to last_obs …`     | `[data.observations]` times                                    | no             |
| `pfilter --fit`                | `[data]`                                                       | no             |
| `survey --fit`                 | `estimate` bounds, `[data]`, `fixed`, `scenario`, anchors      | no             |
| `profile --fit`                | `estimate` priors and bounds, `fixed`                          | no             |
| `fit predict`, `compare`       | via the run's archived config; `stages` only to pick the cloud | terminal stage |

Verified by reading each call site listed in
`rg -n 'FitConfigV2::load' rust/crates/cli/src`. Five of six readers want the
problem and are charged for the pipeline.

### 1.5 Three frictions from one migration

A three-province model migrated to the observation-window format (camdl
`0.1.0+e44f04a9`, data vintage `2026-09-04-r0`) reported three frictions, one of
which cost a six-hour fit. Each is verified here; each is a requirement on §3.5.

**A prior predictive on the data's own design cannot be produced.** The model's
incidence streams declare `window_start`/`window_stop` columns because 39 of 702
daily rows span two to five days, which no uniform `covers` form can state.
`simulate --obs` refuses a windowed stream outright (a single wide file has one
shared time column). `simulate --obs-dir` writes window columns, but on the
model's `emit_schedule` (`obs_emit_schedule_times`, `main.rs:3260`), and
`simulate --draws prior --fit` reads the config's priors and never its `[data]`
(§1.4). So the only prior predictive available is on a uniform grid, which is a
different design from the one the fit will see: on `every 1 'days` the 39
multi-day rows become runs of one-day rows, and the check has more information
than the fit. The team's workaround was to aggregate to seven-day bins so
`ending_on(time, 7 'days)` could state them, at the cost of 21 rows carrying
4.0% of counts attributed wholly to one side of a boundary. This is gh#831's
finding restated on window columns: the self-consistency test's inputs must be
simulated on the real design, and nothing in the CLI produces them. The
fixed-truth recovery path is worse off: `[synthetic]` writes one wide
`ds_NN.tsv` per replicate (`fit/synthetic.rs:35`) and refuses a windowed stream
outright (`synthetic.rs:217`), so the self-consistency test the docs call the
highest-return step cannot run on this model at all.

**A statically decidable horizon mismatch surfaced after the fit.** The weekly
model — `covers = ending_on(time, 7 'days)`,
`simulate { to = last_obs + 8
'weeks }` — fitted for six hours (sixteen chains,
2,000 sweeps), then `fit predict` refused: the forecast row labelled 216 closes
at 217 under `ending_on`, one day past the horizon of 216, and a period boundary
must be a recorded output time. Every input to that arithmetic is fixed once
data are bound: the observed times, the horizon `last_obs` resolves to, the
`covers` form, the output schedule. The check lives in `project_coverages`
(`main.rs:3411`), which asks whether a boundary is a recorded snapshot of a
materialised trajectory, so it cannot run until one exists. The fix — one day of
horizon — re-keys the model and orphans the fit.

**A partial predictive is discarded whole.** When the free-forward tail fails,
`fit predict` refuses the entire artifact by default, including the one-step
half, which is data-conditioned, never reaches the horizon, and was complete.
`--allow-missing-free-forward` recovers it. The team did not ask for a change
and endorsed fail-closed; the cost was an empty prediction tab for a reason
unrelated to the object it hid.

How `fit predict` builds its rows, verified because §3.5 depends on it: the
observed prefix takes each bound row's own period from the loader
(`StreamTimes::coverage`), so per-row windows are already reused exactly; the
forecast tail continues the modal gap of the observed labels across the snapshot
grid (`forecast_times`, `predict.rs:2728`) and gives each label its period from
the declaration — `Covers::period_of` for a uniform form, and for window
columns, which have no rule to continue, the contiguous reading
`[previous stop, label)` (`leaf_row_coverages`, `predict.rs:2642`). A windowed
stream's tail therefore closes at its label and cannot overrun the horizon; a
uniform form with a closing offset can, which is the second friction.

---

## 2. The problem in one sentence

`[stages]` conflates three things — _which algorithm runs_, _where its chains
start_, and _a pipeline over several of them_ — and it is the third that
produced every measured defect: chaining is the only composition primitive it
offers, chaining is a point warm-start, and a point warm-start makes the
multi-chain diagnostic vacuous while the provenance record describes it as
spread.

The fix is structural rather than a patch to `init_mle` because the harm is not
in one key. A pipeline inside a config file needs ordering (the `IndexMap`),
dependency validation (the DAG), cross-step policy (two gates and an override
flag), a rule for which step's output is "the" result (the terminal-stage search
in `posterior_draws.rs:102`), and a story for every non-fit reader that must
skip all of it. Each of those is code that exists today for one feature, and the
feature's sole justified use — priming PGAS with an IF2 point — is no longer
needed in practice and is harmful when used.

---

## 3. Design

### 3.1 Types

```rust
/// The inference problem: what is estimated, from what, on which model.
/// Every command that accepts `--fit` takes `&Problem` and nothing more.
pub struct Problem {
    pub model: ModelRef,
    pub data: Option<DataSpec>,             // exactly one of data / synthetic
    pub synthetic: Option<SyntheticSpec>,
    pub estimate: IndexMap<String, EstimateSpecV2>,
    pub fixed: FixedParams,
    pub simplex_groups: Vec<SimplexGroup>,
    pub scenario: Option<String>,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
    pub config: FitBackendConfig,           // dt, obs_alignment, allow_degenerate_rates
    pub ic_free: Option<bool>,              // conditions the estimand; validated per method
    pub output_dir: Option<String>,
    pub provenance: Option<FitProvenance>,
}

/// One way of fitting the problem. A file carries at most one.
pub struct Method {
    pub algorithm: Algorithm,               // today's `Stage`, minus the three start fields
    pub starts: ChainStarts,
}

/// The inference half: what only `fit run` reads. `None` is a complete
/// problem with no method, which every non-fit reader accepts and
/// `fit run` refuses by name.
pub struct Inference {
    pub method: Option<Method>,             // `[method]`; no map, no order, no chaining
    pub fit_seeds: Option<Vec<u64>>,
}

/// The file. Parsed flat (serde's `flatten` is incompatible with
/// `deny_unknown_fields`), then split. `Problem::load(path)` is the entry
/// point for every non-fit reader; it discards the inference half.
pub struct FitConfig { problem: Problem, inference: Inference }

/// Where a method's chains begin. The top-level split is the fact a reader
/// of `fit summary` needs: are the starts a distribution (one draw per
/// chain) or a point (every chain at one vector)? R̂ is informative only
/// for the first.
pub enum ChainStarts {
    Spread(Spread),
    Point(Point),
}
pub enum Spread {
    UniformUnconstrained,                              // default (Stan's init radius)
    Lhs,
    Uniform,
    FromPrior,
    FromPosterior { source: Handle },                  // one posterior row per chain
}
pub enum Point {
    Declared,                                          // `[estimate].start` / model value; wire "single"
    FromMle    { source: Handle },                     // the escape hatch
    FromParams { path: PathBuf },
}
```

`Algorithm` is `Stage` with `starts_from`, `init_method`, `survey_path`, and
`survey_top_k_n` removed from every variant; nothing else about the eight
variants changes. `StartsFrom`, `FitStarts`, and `InitMethod` are deleted;
`ChainStarts` is `InitMethod` regrouped so that the point-versus-spread fact is
a variant, not a property one has to know per mode. `survey_top_k` is not
carried over: `survey` is little used, and the book's version of the same move —
many short chains to find the modes, then fewer chains started from what was
found (§12.3) — is `from_posterior` from a short run.

`Handle` resolves through `FitRef::classify` for fit sources and accepts a draws
TSV (`from_posterior`) or a params TOML (`from_params`) path directly, as today.
Identity is unchanged in kind: the resolved source's content digest folds into
the method leaf's `deps`, exactly as `InitMethod::source_file` and
`cas_dep_from_dir` do now (`init.rs:133-149`), so rewriting a source in place
re-keys the run.

### 3.2 The file, before and after

Before (today's `docs/fit-toml.md` example, abridged):

```toml
[estimate]
beta = { bounds = [0.001, 0.5], prior = { log_normal = { mu = -2.0, sigma = 1.0 } } }
gamma = { bounds = [0.01, 1.0], prior = { log_normal = { mu = -1.2, sigma = 0.5 } } }

[stages.scout] # runs first
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 2000
iterations = 150
cooling = 0.7

[stages.posterior] # runs second; every chain starts at scout's MLE
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 600
sweeps = 300
init_mle = "scout"
```

After:

```toml
[estimate] # bounds and priors live in the model (gh#369)
beta = {}
gamma = {}

[method] # the fit
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 600
sweeps = 300
# starts = "uniform_unconstrained"   (default: one independent draw per chain)
```

The comparator is a second file with the same problem half — `fit-if2.toml`,
say, with `[method] algorithm = "if2"` — and the two share the fit-level hash
(computed from the problem alone) in sibling segments labelled by file stem, so
each resolves by its own handle. A file with no `[method]` at all is a complete
problem and loads for every non-fit reader. Warm starts, when wanted, are
written where they are used and say what they are:

```toml
[method]
algorithm = "pgas"
starts = { from_posterior = "@base" } # spread: one draw of @base's posterior per chain
```

```toml
[method]
algorithm = "pgas"
starts = { from_mle = "@mle" } # point: every chain at @mle's estimate; R̂ not assessable
```

The wire form of `starts` is a bare string for parameterless rules
(`"uniform_unconstrained"`, `"lhs"`, `"uniform"`, `"from_prior"`, `"single"`)
and a one-key inline table for sourced rules (`{ from_posterior = … }`,
`{ from_mle = … }`, `{ from_params = … }`). One key for one concept replaces
four (`init`, `init_mle`, `survey_path`, `survey_top_k_n`) plus the dead
`fit_starts`.

### 3.3 Commands

```
camdl fit run fit.toml --label base
camdl fit summary @base
camdl fit predict @base
camdl fit new --from fit.toml fit-if2.toml       # copy with provenance; edit [method]
camdl fit run fit-if2.toml --label mle
camdl compare @base @mle
```

`fit run` runs the file's one `[method]` and refuses a file that has none,
naming the table. `--starts <spec>` overrides the method's `starts` with the
same spellings the file uses (`--starts from_prior`,
`--starts from_posterior=@base`): the grammar is a name, or a name and a handle,
replacing the four flags `--init`, `--posterior`, `--mle`, `--params`;
`--survey-path` and `--survey-top-k` are deleted with the mode. Every `--stage`
flag (`fit summary`, `fit predict`, and the `fit run` flags that require one)
goes: a handle names one leaf, so there is nothing to select. The "terminal
Bayesian stage" search that only declaration order could define
(`posterior_draws.rs:102`) goes with it.

`fit new` today copies the source file and injects a `[provenance]` block naming
it; the derived file's `[method]` is then edited by hand. A
`--method <algorithm>` flag that rewrites that one table is a follow-up (§9), so
the comparator is derived rather than retyped.

The two cross-stage gates go with chaining. Gate 1's purpose survives in one
place: resolving a `from_mle` or `from_posterior` handle whose stored verdict is
not converged is refused unless `--allow-nonconverged-source` is passed, which
is today's `--allow-nonconverged-scout` moved to the seam where the upstream is
consumed. Gate 2 (a refine must not regress below its scout) is dropped: it
guarded an in-process handoff that no longer exists, and across invocations the
regression is visible in `fit table`.

The workflow verbs, all under `fit`, all taking the config positionally the way
`fit run` does:

```
camdl fit preflight   fit.toml [-n 200] [--seed S]
camdl fit recovery    fit.toml [--replicates R] [--truth theta.toml]
camdl fit calibration fit.toml [--simulations S] [--reject <rule>]   # reserved; not built
camdl check model.camdl                                              # unchanged: the camdlc type-check
```

`check` today forwards its arguments verbatim to `camdlc check` (`main.rs:210`,
`Passthrough`) and stays that: a static check of a model file. Housing the
workflow verbs under it was considered and rejected on three grounds. It would
mix static analysis with verbs that draw from priors and simulate; it would need
a positional fallback under which a model file named `prior` must be given with
a path prefix; and `fit preflight` writes the same artifact family `fit predict`
writes (§3.5), so the two are siblings and belong in one namespace. The
organizing rule is that everything taking a `fit.toml` lives under `fit` and
takes it positionally — an earlier shape had the config as `--fit` on the checks
and positional on `fit run`, two spellings of one argument.

### 3.4 Chain starts: spread, point, and what the summary says

The default is `from_prior` whenever every estimated parameter declares a prior
— drawn from the prior the fit scores against, a `[estimate].prior` in the file
over the model's `~` declaration — and `uniform_unconstrained` — an independent
draw per chain on the unconstrained scale (Stan's initialization, described at
§11.2, p. 195) — otherwise. The reason is measured (gh#876): a bounds-uniform
draw at province scale is routinely a start the bootstrap filter cannot score,
because a fixed relative error in a rate is a standardised residual that grows
with the square root of the population, and wide bounds carry no scale. A prior
does. A chain whose start cannot be scored is still refused (gh#887 records the
bounded-retry follow-up); the default just stops manufacturing the case. The two
spread warm-starts are the ones the workflow text endorses. Starting from
posterior draws is the book's own recommendation for a hard posterior — run an
approximate algorithm first and use its draws to initialize ("one way to obtain
good starting points for HMC is to first run a variational algorithm to get near
the typical set", §11.2, p. 196; the same move with Pathfinder in §12.3, p. 218:
run many chains from different initial values to find modes, then start fewer
chains from the found modes). `from_prior` constrains starts "to be within a
reasonable region as determined by the prior", which §12.5 (p. 234) lists as the
legitimate refinement of an initialization scheme once the geometry is
understood. Both keep one independent draw per chain, so the between-chain
comparison R̂ makes is still a comparison.

A point start is kept as an explicit escape hatch — it is the right tool for
continuing an optimizer, for reproducing a specific run, and for a deterministic
test — and it is made honest in three places:

1. `fit_state.toml` gains `chain_starts_kind = "spread" | "point"` beside the
   existing `chain_init_source`, and `fit summary` prints the one-line header
   that `state.rs:131` has promised since the field was added. This closes the
   open item from the provenance fixes.
2. When `chain_starts_kind = "point"` and the method reports an R̂,
   `RhatBand::NotAssessed` (`method_result.rs:575`) is the band for every
   parameter, with the reason rendered where the glyph would be:
   `R̂ — not
   assessed: all 4 chains started at one point (starts = from_mle @mle)`.
   The convergence leg of the verdict is therefore _not assessed_, never a pass.
   A `NotAssessed` R̂ is already rendered as `—` and described in words; this
   adds the reason and the trigger.
3. Optimizer-only methods (`nl-sbplx`, `nl-bobyqa`) report no R̂ and are exempt.
   IF2 reports Â, a chain-agreement statistic with the same logic (chains that
   began together are more likely to agree); Â is annotated with the same header
   but keeps its band, because IF2's per-chain perturbation re-spreads the swarm
   after the start. This is a weaker guarantee than spread starts give and the
   header says so.

Nothing refuses a point start; writing `from_mle` in the config or on the
command line is the acknowledgement. A refusal-plus-override would push the same
choice into a flag nobody records.

### 3.5 What the checks compute, what they fail on, and what they never say

The verbs share one report type:

```rust
pub struct CheckReport {
    pub kind: CheckKind,                    // Preflight | Recovery | Calibration
    pub problem: ContentHash,               // the fit-level digest
    pub n: usize, pub seed: u64,
    pub failures: Vec<DeterministicFailure>,   // non-empty ⇒ exit status 1
    pub frequencies: Vec<Frequency>,           // reported; no threshold, no glyph
    pub verdict: Verdict,                      // the only value is NotAssessed
}
pub enum DeterministicFailure {
    EvaluationFailed  { draw: usize, reason: String },       // a rate, integrator, or density refused
    NonFinite         { draw: usize, at: Site },             // NaN or ±inf in a state or observation
    StateOutOfRange   { draw: usize, at: Site, value: f64 }, // a compartment outside [0, N]
    PriorNotSampleable{ param: String, reason: String },     // flat, degenerate, or absent
}
pub struct Frequency { pub name: &'static str, pub numerator: usize, pub denominator: usize }
```

The line between `failures` and `frequencies` is not "model-internal versus
about the world"; it is _deterministic versus frequency_. A deterministic
failure is a fact about the model as written that holds for the draw that
produced it regardless of how often it occurs: a non-finite value, an
observation density that cannot be evaluated, a latent state outside its
physical range, a prior that cannot be sampled. One occurrence is a bug, so the
check stops and says which draw. Everything else is a frequency — the fraction
of draws in which a reported count exceeds the population, in which a stream is
identically zero, in which the epidemic infects more than nine tenths of the
population by day 30 — and a frequency has no threshold that camdl can set,
because the threshold is a statement about the population and the surveillance
system, not about the model. The book draws the same line: "there is also no
universal way to decide when a check fails or whether failure requires
adjustments to the model" (§8.2, p. 143), and, of the workflow as a whole,
"model checking … is not automated nor could it be automated. However steps like
model fitting and comparison can be partly automated to aid analysts" (§2.2, p.
20). So camdl automates the computation and the counting and stops there.

This differs from the candidate rule that listed "counts exceeding the
population" among the automatic failures. For a chain-binomial process the
latent state cannot exceed the population, so that case is already
`StateOutOfRange` on the ODE backend and impossible elsewhere. A _reported_
count under a negative-binomial observation model can exceed the population,
because that family's support is unbounded; whether 2 of 200 prior draws doing
so is a problem is a judgement about the reporting model, and it belongs in the
frequency table where the human reads it. An all-zero stream in every draw is
the same: a frequency of 200/200 is loud without a threshold, and if the stream
is _structurally_ zero the static lint (L402, dead compartment) is the right
detector.

**`verdict` has one value.** The report's exit status is 0 when it computed and
1 when a deterministic failure occurred; there is no status for "passed". A
prior predictive that computes cleanly prints its frequency table and the line
`verdict: not assessed — compare the bands to what is plausible for this
population`,
and its JSON carries `"verdict": "not_assessed"`. A check whose green glyph can
be read as validation is a machine for false greens, and on a surface that
informs public-health decisions a false green is the worst output available.

**`fit preflight`.** Everything a modeller should know before spending the
compute, in one report with four blocks. The first three are static — functions
of the problem and the method, no simulation — and are the part that answers
"how much am I about to run, and can it complete":

- _The observation design._ Per stream: the number of scored time points, the
  coverage windows, the first and last scored boundary, and the horizon. For a
  particle method this is the number of filter steps every particle must
  survive, which is the quantity that has limited success on the province
  models.
- _The budget._ `particles × scored points × chains × sweeps` for the method as
  configured, and a wall-time estimate from a timed one-sweep probe on one
  chain.
- _The forecast horizon._ The rows `fit predict` will emit are a pure function
  of what is bound: the observed labels, the output schedule, and the stream's
  `covers`. Preflight builds them the way predict does — `forecast_times`
  continues the modal observed gap, `leaf_row_coverages` assigns each label its
  period — and reports, per stream, how many forecast rows there are and the day
  the last one closes. When that day lies past `simulate.to`, as it does for a
  week-ending window whose label is the horizon itself, the block says the
  integration will run to that day. Nothing is refused: the declared `to` is
  what the modeller asked to forecast, and closing the last window is camdl's
  arithmetic, not theirs (§3.5, `fit predict`).

The fourth block is the prior predictive. It draws `n` parameter vectors from
the priors with the existing precedence (fit-toml prior, then the model's `~`
declaration, then an error, `priors_precedence.rs`), simulates each on the
problem's observation design — the bound streams' observation times, coverage
windows, missing-value placement and covariate columns when `[data]` is present
(the design-preserving simulate of gh#831), the model's `emit_schedule`
otherwise — and writes a `preflight` leaf in the store:
`predictive/<stream>.tsv` in the columns `fit predict` writes with
`horizon = prior` and the convergence columns empty, `observed/<stream>.tsv`
when data are bound, the calendar sidecar, and `report.json` carrying all four
blocks. That is the artifact family gh#711 asks for, put where a store walk can
find it. The frequency table is per stream: observed total against the prior
5–95% band of totals, observed peak against the band of peaks, the
count-exceeds-population fraction, the all-zero fraction, and the
bounds-rejection fraction the prior sampler already reports (`main.rs:3903`).
This is the book's test-summary comparison, `T(y)` against `T(y_rep)` (§8.2, p.
142), with no tail probability attached.

The prior predictive is simulated on the bound data's own rows — each stream's
observed times and each row's period exactly as the loader bound them
(`StreamTimes`), so a per-row window, an `NA` hole, and a covariate column are
all reproduced — and the artifact is per stream, so the single-wide-file refusal
in §1.5 never arises. This is the design-preserving simulate gh#831 requests;
`fit preflight` is its verb, and `fit recovery` shares the primitive with
`--truth` in place of prior draws (§8, item 16). For a stream whose likelihood
reads no data column — every stream in the §1.5 model — the block needs only the
loader's `StreamTimes` and the existing emitters, so it can land ahead of
gh#829; a stream with covariates waits on it.

`fit run` runs the three static blocks and the prior predictive's deterministic
checks before any sampling, for any method whose `requires_priors()` is true
(`config_v2.rs:1393`: pgas, pmmh, mh, nuts), with `n = 200` and the fit's seed,
and aborts on a deterministic failure — the "fit fast, fail fast" principle of
§12.1 (p. 209) applied at the cheapest possible point. The leaf is its own kind,
not part of the fit leaf, so the fit's `run_id` is unchanged by whether the
check ran; `fit.meta.json` records
`preflight = <run_id> | skipped | not_applicable`, and `--no-preflight` sets
`skipped` for CI smoke fits. `fit summary` prints the frequency table as a block
under the verdict. A second method on the same problem is a cache hit on the
same leaf.

**`fit recovery`.** The fixed-truth self-consistency test the docs already call
the highest-return step (`docs/diagnosing-fits.md:21`) and the recovery harness
runs by hand (`tests/recovery/README.md`). It reads the `[synthetic]` block
(`true_params`, `sim_seeds`, its own `backend`) — `--truth` and `--replicates`
are the CLI overrides — simulates each replicate on the real design, fits each
with the file's `[method]`, and writes the `coverage.tsv` the 2026-04-17
synthetic-replicates proposal specified and gh#154 still owes: per parameter,
truth, posterior median, 50% and 90% intervals, covered (0/1), and
`z = (median − truth) / sd`. Frequencies are the coverage rates at 50% and 90%
over the replicates; alongside them, per parameter, the ratio of posterior to
prior standard deviation, printed as a number because the point at which "the
data taught nothing" begins is a judgement. Deterministic failures are a
replicate whose fit did not complete or produced a non-finite summary. The
report says what the book says this test can and cannot show: a single truth
point "will possibly flag gross problems, but it does not guarantee anything"
(§14.1, p. 250), and a posterior that equals the prior "could signal a problem
in design or a bug in model implementation" (§6.3, p. 109). For an optimizer
method (`if2`, `nl-*`) the table carries each replicate's estimate and the
spread of estimates across replicates in place of the intervals, and no coverage
column.

Each replicate is written as the files the loader reads: one per stream, under
the stream's declared column names, on the bound rows' own periods — the round
trip gh#833's emitters already make for `simulate --obs-dir`
(`obs_emit_declared.rs`), applied to the real design. Today's `[synthetic]`
writer produces one wide file per replicate and refuses a windowed stream; it is
replaced by the per-stream emitter, so a synthetic dataset loads under the model
that generated it with no hand step between generation and fit. That property —
generate, write, load, fit, with the same loader on both sides — is what makes
`fit recovery` a test of the model rather than of a transcription.

One config rule stands in the way and is relaxed for this verb. Today `[data]`
and `[synthetic]` are mutually exclusive (`FitConfigV2::validate`), so a
`[synthetic]` config binds no data and can only simulate on the declared
`emit_schedule`. Recovery on the real design needs both halves at once: `[data]`
supplies the observation design — the rows, their periods, the holes, the
covariate columns — and `[synthetic]` (or `--truth`) supplies the parameter
values; the observed _values_ in `[data]` are read for their shape and never
scored. The problem half therefore admits `[data]` beside `[synthetic]`, and
`fit run` on such a file fits the real data as it always did while
`fit recovery` fits the replicates (§8, item 19).

**`fit calibration`** is a reserved slot: the verb name and the artifact
(`ranks.tsv`, one row per simulation and parameter, the rank of the drawn truth
among the posterior draws) are fixed here so nothing else takes them, and the
verb is not built in this proposal's increments. Classical SBC — `S`
simulations, each a prior draw, a simulated dataset on the design, a fit with
the file's `[method]`, and a rank — is specified as a follow-up (§9), with the
`γ` statistic and the data-only rejection rule the book sanctions.

**Posterior predictive checking is `fit predict`**, which already emits the
banded family with `free_forward` and `one_step` horizons. It gains the same
deterministic-failure rules and frequency table, and one label: every predictive
artifact records `evaluation = in_sample | held_out` from the fit's holdout
declaration, and the summary prints it. It is not computed automatically. The
posterior predictive is the check most easily read as validation, and the book
is direct about why: "as it uses the same data for model fitting and misfit
evaluation, it can be overly optimistic or misleading. That is, a miscalibrated
model can still produce reasonable predictions because of overfitting" (§8.3, p.
147). An always-on posterior predictive that printed a coverage number under
every fit would be read as a pass rate; the honest predictive number is the
held-out one, which the 2026-08-29 honest-predictive-evaluation proposal stages.
What `fit summary` does instead, when no predictive artifact exists for a fit,
is print the one command that would make one. A nudge is not a verdict.

Two rules from §1.5 attach here. `fit predict` keeps reusing the bound rows' own
periods for the observed prefix and the declaration's continuation for the tail,
as it does today; the proposal changes nothing about which windows a predictive
row covers. What changes is where the integration stops. Predict runs each draw
to the close of the last forecast window rather than to `simulate.to`, and
records that close as an output time, so a row whose window ends past the
declared horizon — by up to one span: a day under `ending_on(time, 7 'days)`, a
week under `starting_on(time, 7 'days)` — is scored instead of refused; the
summary says to what day and why. The declared `to` still means what it says —
how far to forecast — and the model is unchanged, so nothing re-keys. This is
the six-hour case in §1.5: the fit was never at fault, and predict now closes
the window it was asked for. And when the free-forward tail fails
deterministically — a boundary off the schedule, an unresolvable horizon — the
one-step artifact, which is data-conditioned and never reaches the horizon, is
written, the failure is recorded in `report.json` under `failures`, and the exit
status is 1. That is the `CheckReport` contract applied to predict: compute what
is computable, name what failed, pass nothing. `--allow-missing-free-forward` is
then the default's record rather than a flag, and a prediction tab is never
empty for a reason unrelated to the object it hides (§8, item 15).

### 3.6 Where orchestration lives

The question is whether a small TOML that runs a few verbs in order — check,
fit, predict — belongs inside camdl for the trivial case, with Snakemake or
`make` reserved for fan-out. It does not, and the reason is the same one that
makes the checks refuse to pass.

A pipeline earns its keep by running unattended. A check earns its keep by
stopping for a human. Any sequence that contains a check therefore does one of
two things at that step: it stops and waits (in which case it is a script the
human runs a line at a time, and `&&` already expresses it), or it decides (in
which case it is the auto-pass this design refuses). There is no third behaviour
for a pipeline to implement. The `[stages]` DAG was the first version of exactly
this engine — ordering, dependencies, a gate with an override flag — and its one
composition primitive produced findings 1 and 2.

The test that tells the two apart, stated so an implementer can apply it to the
next request: **a sequence enters camdl as one verb if and only if no step
between its parts requires a judgement camdl cannot make.** `fit recovery`
passes the test — simulate at `θ*`, fit, compare to `θ*` — because every step is
a function of declared inputs and the comparison is arithmetic.
`fit calibration` passes it. `fit run` followed by `fit predict` passes it too,
and that composite is the `--predict` a later increment may add to `fit run`.
"Check the prior, then fit if it looks right" fails the test at "looks right";
"fit twelve provinces with three samplers and compare" fails it at fan-out,
which is a scheduling problem and not a statistical one.

What the trivial case actually needs, camdl already provides, and the design
keeps: every verb is idempotent and memoized by the content-addressed store (a
second `fit run` on an unchanged config is a cache hit, `fit/mod.rs:1175`; a
second `simulate` prints `cached:`), so a shell script of verbs is already an
incremental build without a DAG; every artifact is addressable by handle, so a
script never carries a path; and the checks' exit codes and `--json` give a
script something to branch on for the deterministic failures, which are the only
branch a script may take on its own. The workflow text lands in the same place:
it recommends "expressing all steps within scripts" and names Snakemake and
`targets` as the tools "to improve reproducibility of the simulation or data
analysis workflow steps, while recomputing only the required steps" (§15.4, p.
258). `docs/workflow.md` gains a worked `Makefile` for the four-verb case so
that the recommendation is a file someone copies, not advice.

---

## 4. Options considered

**A. Keep `[stages]`; make `init_mle` a spread start.** Chained chains would
draw from the upstream's posterior instead of sitting at its point. Fixes
finding 1 and nothing else: the pipeline semantics, the DAG, both gates, the
terminal-stage rule, and the mandatory block on non-fit readers all stay, and
there is still nowhere for a workflow step to go. The cheapest option and the
one that leaves the shape that produced four provenance defects in place.

**B. Remove the inference block entirely; every method knob on the command
line.** The CLI already folds `--init` and its companions into stage identity
before the store claim (`fit/mod.rs:327-368`), so identity would survive. What
would not is the record: the method a fit used would live in shell history, and
two methods of one problem would have no shared document. It also contradicts
the book's reproducibility advice (§15.4) and the existing rule that a
`fit.toml` is a bundle of flags, not a gate for them.

**C. Two files: `problem.toml` and `method.toml`.** The cleanest type boundary,
and the one Snakemake users would like. Costs: every handle that resolves a
config would need two paths; relative-path anchoring, which the config already
gets right, would have two anchors; and `fit new`, `fit diff`, and
`config_identity_hash` would each grow a second input. The type split in §3.1
gives the same guarantee at the function signature (`&Problem`) without the
file-system cost.

**D. One file, two halves, one `[method]` (recommended).** A mechanical rename
for unchained configs; the store's own factoring; no ordering, so nothing to
validate; the problem is a complete document on its own; a handle names one
leaf, so no verb needs a `--method`. A named map of methods in one file
(`[methods.<name>]`, one selected per invocation) was the earlier form of this
option. It buys the problem half being written once, and costs a `--method` flag
on every verb that takes a config, a search in `fit predict` for the one leaf
holding a posterior cloud — the terminal-stage rule under a new key — and a
shape that reads as a pipeline after the text says it is not. The common case is
one method; the comparison case is a second file; the two share the fit-level
hash in sibling segments.

**E. Workflow steps as config entries — a `[checks]` table or new `Stage`
variants.** Every check's parameters (`n`, seed, rejection rule) would enter the
fit-level or stage-level identity, so editing a check re-keys a fit; and it
reintroduces the question of which entries run when, which is the pipeline again
under a different key. The verbs in §3.3 keep check parameters on the check's
own leaf.

---

## 5. Migration

**What breaks.** Every `fit.toml` with a `[stages]` block. The loader rejects it
with the rewrite spelled out, in the pattern the legacy `starts_from` /
`init_method` keys already use (`validate_stage_keys`, `config_v2.rs:2240`):

```
error: legacy table `[stages.posterior]`
  replacement: rename to `[method]` and run it with
    camdl fit run fit.toml
  a file carries one `[method]`; put `[stages.scout]` in its own file
  `init_mle = "scout"` has no replacement in the file: it started every chain at
  scout's point estimate, which makes R̂ uninformative. Run scout first and, if a
  warm start is wanted, write one of
    starts = { from_posterior = "@scout" }   # one draw per chain (keeps R̂ meaningful)
    starts = { from_mle = "@scout" }         # every chain at one point (R̂ not assessed)
```

No silent conversion and no compatibility path, per the alpha posture. The
mechanical half (rename the table, delete `init_mle`, `fit_starts`,
`survey_path`, `survey_top_k_n`; fold `init` into `starts`) is done once by a
script over the repository's own six committed fixture configs
(`git ls-files '*.toml' | xargs grep -l '\[stages'`: the `mre` and
`polio_afp_es` fixtures and the four `tests/recovery` cases); the semantic half
— a chained stage — is a decision the message hands to the author, because the
file cannot name a handle that does not exist until the upstream has run.
In-repository, chained configs are rare: `init_mle` appears in no committed
`.toml`, and in eleven Rust files (three integration tests, the rest doc
comments and unit tests).

**What re-keys.** Two levels, both deliberately.

- _Fit level._ `fit_starts` is `Option<FitStarts>` with `#[serde(default)]` and
  no `skip_serializing_if`, so it serializes as `"fit_starts": null` into the
  canonical JSON that `fit_config_blob_hash` digests (`fit/cas.rs:422`;
  `serialize_minus` removes only the three named keys, `fit/cas.rs:224-246`).
  Deleting the field changes every fit-level digest. Every stored fit is
  therefore a cache miss after this lands; pre-1.0, that is the sanctioned cost
  of not carrying a dead field, and it is stated here so it is confirmed rather
  than discovered.
- _Method level._ `Stage::identity_payload` serializes the whole stage minus its
  extension dimension (`config_v2.rs:1473-1511`), so `init`, `init_mle`,
  `survey_path`, and `survey_top_k_n` are in every stage leaf's hash today.
  Regrouping them into `starts` changes the payload shape and re-keys every
  method leaf, including ones that never used a warm start.
- _Not re-keyed._ `sim`, `pfilter`, `survey`, `profile_point`, and
  `sim_ensemble` leaves, because `LEVEL_SCHEMA_VERSION` (`fit/cas.rs`) is not
  bumped: the content change re-keys the two affected kinds by itself, and a
  schema bump would invalidate four kinds for a change that touches none of
  them. The new digests are pinned by a `run_id`-stability test as the runid
  crate doc requires.
- _Path labels._ The fit level keeps its name; the `stage` level is renamed
  `method` and loses its `NN-` ordinal prefix (`01-posterior-<h8>` becomes
  `posterior-<h8>`), because the ordinal encoded execution order and there is
  none. `run_id` hashes level hashes only (`runid/src/kind.rs:79-88`), so this
  is a label change; `docs/dev/cas-path-shape-contract.md` gains a migration
  line.

**Blast radius, counted.** 67 integration-test files under
`rust/crates/cli/tests` contain `[stages` (107 occurrences); `config_v2.rs` has
116 embedded configs in its unit tests; 42 files under `docs/` reference
`[stages`; four of those (`docs/workflow.md`, `docs/inference.md`,
`docs/debugging.md`, `docs/diagnosing-fits.md`) are parsed by
`make test-cli-docs` (`Makefile:220`), which fails the gate on any command line
that no longer parses, so the doc rewrite lands in the same change as the flag
rename. The `camdl-book` chapters that teach the scout → posterior idiom are
outside this repository and are the first external migration. `fit new`'s hint,
which still says `starts_from` (gh#593 item 2), is rewritten to name `starts`.

**Kept as they are.** `[synthetic]` (it is a data source and is
identity-bearing); `fit_seeds` (already excluded from the fit-level hash);
`ic_free` and the `[estimate]` keys `perturb_only_at_t0` and `rw_sd`, which are
IF2 schedule knobs living in the problem half. Today the last two are a
load-time error when no `if2` stage exists; under one method per file they are
inert for a non-IF2 method and the loader says nothing, so that a problem's IF2
comparator file and its PGAS file differ only in `[method]` and share a
fit-level hash. Relocating them under `[method]` is a follow-up, not a
precondition. The `dangling_priors_warning` ("priors declared but no stage uses
them", `config_v2.rs:2461`) is deleted: with one method per file, priors an
optimizer ignores are the normal case.

---

## 6. Increments

Each lands as a reviewable unit with the full gate green and leaves the tree
strictly more honest than before it.

1. **The split and the starts.** `Problem` / `Inference` / `Method` /
   `ChainStarts`; `[method]`; `--starts`; the `[stages]` rejection message;
   deletion of `StartsFrom`, `FitStarts`, `InitMethod`, `survey_top_k`, the
   stage DAG, every `--stage` flag, both gates and `--allow-nonconverged-scout`;
   `--allow-nonconverged-source` at handle resolution; `chain_starts_kind`, the
   summary header, and `RhatBand::NotAssessed` for point starts; the `run_id`
   stability pins; the path-shape contract; the four `test-cli-docs` documents.
   This is the re-keying increment.
2. **`fit preflight`.** The three static blocks (design, budget, forecast
   horizon) land first: they are functions of the problem, the method, and the
   bound observation times, built on `forecast_times` and `leaf_row_coverages`
   as they stand. The prior predictive block lands next for streams whose
   likelihood reads no data column, on the loader's `StreamTimes` and the
   existing emitters; for streams with covariates it waits on gh#829
   (covariate-conditioned streams simulate as zero) and the covariate half of
   gh#830 (its header half landed with gh#833). Lands the `CheckReport` type,
   the `preflight` store kind, the `fit run` pre-flight with `--no-preflight`,
   and the summary block. gh#711 and gh#831 close with it.
3. **`fit recovery`.** The `coverage.tsv` roll-up (gh#154's remaining piece).
   The `tests/recovery` harness becomes a `fit recovery` invocation per case.
   `fit calibration` is reserved, not built (§9).
4. **`fit predict`'s failure rules and evaluation label**, sequenced with Stage
   3 of the honest-predictive-evaluation proposal so that `held_out` is written
   only when a holdout was applied.
5. **Docs**: `fit-toml.md`, `workflow.md` with the worked `Makefile`,
   `agents.md`, `inference.md`, the run and inference specs; then the book.

---

## 7. Tradeoffs stated plainly

- A user who liked one command running scout and posterior now types two, and
  the second names the first by handle. That is the cost of making the warm
  start visible; it is one line.
- Every stored fit becomes a cache miss once. Alpha posture accepts this; it is
  flagged in §5 so it is a decision, not a surprise.
- A comparator is a second file whose problem half duplicates the first's. A
  change to one that is not made to the other is a different problem, and the
  fit-level hash is what says so; `compare`'s bound-data preflight (gh#713)
  already refuses to compare fits of different data.
- The pre-flight adds seconds to every prior-consuming `fit run` and minutes to
  a very large spatial model; `--no-preflight` exists and is recorded.
- The `[estimate]` block keeps two IF2 knobs it should not own, for one more
  increment.

---

## 8. Decisions for the maintainer

Each with a recommendation and a confidence: _solid_ (would be surprised to be
wrong), _leaning_ (a reasonable person could choose the other way), _need-you_
(the call is a policy the maintainer owns).

1. **One `[method]` per file, no map, no order and no chaining** (§3.1–3.3).
   _Ruled 2026-09-09._ The named map was the earlier form; the maintainer's
   objection — that the common case is one method and a comparator is a second
   file — removes the `--method` flag, the `fit predict` search, and the
   pipeline-shaped map in one move.
2. **Delete in-file chaining; keep `from_mle` as an explicit handle-sourced
   point start; report R̂ as not assessed for point-started samplers** (§3.4).
   Recommend as written. _Solid._ The measured collapse and the book's stated
   purpose for dispersed starts (§30.5, p. 469) leave no version of chaining
   that is both automatic and honest.
3. **Fold `--init`, `--posterior`, `--mle`, `--params` into `--starts <spec>`;
   delete `--survey-path` and `--survey-top-k` with `survey_top_k`** (§3.3).
   _Ruled 2026-09-09._ Four flags for one concept is the shape that let `init`
   and `init_mle` disagree; with the survey mode gone the grammar is a name or a
   name and a handle.
4. **The workflow verbs live under `fit`; `check` stays the static model check;
   `fit preflight` absorbs the prior predictive and the design, budget and
   horizon checks** (§3.3, §3.5). _Ruled 2026-09-09._ Everything that takes a
   `fit.toml` lives under `fit` and takes it positionally; the earlier `check`
   placement mixed static and dynamic analysis and needed a positional fallback.
5. **`fit run` runs preflight's static blocks and the prior predictive's
   deterministic checks before sampling, stored as its own kind, with
   `--no-preflight` recorded** (§3.5). _Ruled 2026-09-09._ The case against is
   cost on very large models; the case for is that an unrun check is the status
   quo the design exists to end, and the forecast-horizon rule in §3.5 is what
   would have saved a six-hour fit.
6. **No separate posterior-predictive verb; `fit predict` is the posterior
   predictive and gains the failure rules and the `in_sample | held_out` label**
   (§3.5). _Ruled 2026-09-09._ One artifact family, two verbs (`fit preflight`
   with `horizon = prior`, `fit predict` with the fitted horizons), both under
   `fit`.
7. **No pipeline TOML inside camdl; the boundary is the judgement test in
   §3.6.** Recommend as written. _Solid._ The trivial case is already `&&` over
   memoized verbs; the non-trivial case is a scheduler.
8. **The failure/frequency line: deterministic facts stop the check, counts are
   printed, nothing passes** (§3.5). _Ruled 2026-09-09._ A reported count
   exceeding the population is a frequency — the observation family's support
   includes it, and a fail on one tail draw of a negative-binomial reporting
   model would fire on legitimate priors. A latent state outside `[0, N]` is a
   deterministic failure.
9. **Re-key the fit and method levels without bumping `LEVEL_SCHEMA_VERSION`;
   rename the level and drop the ordinal in the same change** (§5). Recommend as
   written. _Leaning._ A schema bump is the greppable form of a deliberate
   turnover, but it would orphan four kinds this change does not touch.
10. **An ad-hoc method from flags alone (`--algorithm pgas --chains 4 …`, no
    `[method]` table)** is deferred to a second increment. _Leaning._ It is
    consistent with every-behaviour-a-flag and with identity (overrides are
    already folded before the claim), and it is not needed to land the split.
11. **Gate 1 survives as `--allow-nonconverged-source` at handle resolution;
    Gate 2 is dropped** (§3.3). _Leaning._ Gate 2's regression check guarded an
    in-process handoff; across invocations `fit table` shows the same number.
12. **`perturb_only_at_t0` and `rw_sd` stay in `[estimate]` and become inert for
    non-IF2 methods**, with relocation under the method as a follow-up.
    _Leaning._ Moving them now widens the first increment for a key two fixtures
    use.
13. **`survey_top_k` is not carried into `ChainStarts`** (§3.1). _Ruled
    2026-09-09._ `survey` is little used; `from_posterior` from a short run is
    the book's form of the same move.
14. **SBC is a reserved verb and artifact, not built** (§3.5, §9). _Ruled
    2026-09-09._

15. **`fit predict` writes the one-step artifact when the free-forward tail
    fails, records the failure, and exits 1; `--allow-missing-free-forward`
    goes** (§3.5). _Ruled 2026-09-09._ Fail-closed on the exit status and the
    report; a complete object is no longer discarded for an unrelated reason.
    The flag the §1.5 team named does not exist under that spelling;
    `--horizon one_step` is the existing way to skip the tail, and it stays.
16. **The design-preserving simulate is one primitive with two verbs on it:
    `fit preflight` (prior draws) and `fit recovery` (`--truth`)** (§3.5).
    _Leaning._ gh#831 asked for `simulate --design-from <fit.toml>` as well;
    exposing the primitive there is cheap once it exists and is the shape
    `simulate --draws prior --fit` should have had. Recommend exposing it, so a
    script can produce synthetic data on the real design without a verb's report
    attached.
17. **Preflight's prior predictive ships for covariate-free streams before
    gh#829 lands** (§6). _Solid._ The §1.5 model has no covariate streams, the
    blocked case is exactly the one gh#829 names, and refusing a stream with
    covariates by name until then is the honest partial.

18. **`fit predict` integrates each draw to the close of the last forecast
    window and records that time as an output; no fit-start refusal** (§3.5).
    _Ruled 2026-09-09._ The modeller asked for a horizon; the extra span a
    window that closes past its label needs is camdl's bookkeeping. A refusal at
    fit start was the earlier form; it guarded a failure predict can simply not
    have.

19. **`[data]` and `[synthetic]` may coexist; `fit recovery` reads the design
    from the first and the truth from the second** (§3.5). _Solid._ The
    design-preserving primitive already carries a `Bound` design built from a
    fit's loaded streams; the exclusion in `FitConfigV2::validate` is the only
    thing between it and recovery on the real design. Without this, recovery can
    only simulate on the declared schedule, which is the case gh#831 exists to
    end.

20. **`from_prior` is the default `starts` when every estimated parameter has a
    prior; `uniform_unconstrained` otherwise** (§3.4). _Ruled 2026-09-09._
21. **An unscoreable start keeps refusing the chain; bounded retry of a spread
    start is gh#887, landing with the `starts` type** (§3.4). _Ruled
    2026-09-09._
22. **The split lands; the ebola agents fold the config edit into their
    window-format migration** (§5). _Ruled 2026-09-09._ It lands after the
    proportion-of-flows projection, so the two re-keys ship in one release.

---

## 9. Named follow-ups

- **`fit calibration`, the SBC verb.** Classical SBC: `S` simulations, each a
  prior draw `θ_s`, a dataset `y_s` simulated on the design, a fit with the
  file's `[method]`, and the rank of `θ_s` among the posterior draws (thinned
  toward independence, §14.1, p. 251) written to the `ranks.tsv` reserved here.
  Per parameter, the `γ` statistic of Säilynoja, Bürkner and Vehtari (2022) as
  used in §14.2 (p. 251) with its quantile under uniform ranks — a number, not a
  pass. `--reject <rule>` records a data-only rejection rule in the leaf, which
  the book sanctions for weak priors that simulate implausible datasets "as long
  as the criterion only depends on data (and not on latent parameters)" (§14.3,
  p. 253). Two cautions the report should repeat: SBC "clashes with the common
  practice of specifying wide priors" (§14.3, p. 252), so for a compartmental
  model it may check calibration mostly where no one would fit; and "it is
  better to run a few simulations than no simulations at all" (§14.3, p. 253),
  so `--simulations 8` is a legitimate first run. Then the rank-ECDF difference
  plot with simultaneous bands (§14.2) and posterior SBC (Säilynoja, Schmitt et
  al. 2026) as `fit calibration --from @fit`. A separate proposal.
- **`fit new --method <algorithm>`**, rewriting the derived file's `[method]`
  table so a comparator is derived rather than retyped, and replacing the hint
  that still names `starts_from` (gh#593 item 2).
- **User-declared data-only rejection rules for `fit preflight`**, the
  sanctioned way to encode "an epidemic infecting 90% of the population in three
  days is not a dataset" without camdl judging it (§14.3, p. 253). Needs a small
  expression surface over stream summaries; the `quantities {}` vocabulary is
  the obvious candidate.
- **Relocate `perturb_only_at_t0` and `rw_sd` under `[method]`.**
- **`fit run --predict`**, the decision-free composite of fit and posterior
  predictive that passes the §3.6 test.
- **Chain stacking** as the principled alternative to `--exclude-chains` (§12.5,
  p. 234; Yao, Vehtari and Gelman 2022), which becomes more attractive once
  every fit's chains are known to have started apart.

---

## 10. Found while reading, to file separately

- `fit_starts` is parsed, hashed into every fit-level identity as `null`, used
  once to silence a warning, and drives nothing (`config_v2.rs:72,2468`; no
  reader elsewhere). This proposal deletes it; if the proposal is deferred the
  field should still go.
- `chain_init_source` is documented as surfaced in `fit summary`
  (`state.rs:131`) and is not (§1.3). Closed by increment 1.
- `docs/workflow.md:108` lists `algorithm` as `if2 | pgas | pmmh | pfilter`; the
  enum has eight variants and `docs/fit-toml.md:87` lists all eight.
- `survey` scores every point at a constant `SURVEY_DT = 1.0`
  (`survey.rs:54,447`) and does not read `[config].dt`, so a survey and the fit
  it seeds can run at different discretizations. Not investigated beyond the
  constant. `survey_top_k` leaves `ChainStarts` (§3.1), so the mismatch no
  longer reaches a fit's starts; it still affects `survey` on its own.
- gh#585 says the holdout declarations are inert; `runner.rs:1572` applies
  `holdout_after` and commit `46a5f8bd` documents it as live. The issue is stale
  and likely closable.
- The `[model]` hint in `wrap_fit_load_error` (`main.rs:3644`) is attached to
  every load error on the `--draws prior` path, including one whose cause is a
  missing field the hint does not mention. Fixed for the `stages` case by making
  the field optional; the hint should still be scoped to the "not a fit config
  at all" error it was written for.

---

## References

- Gelman, A., Vehtari, A., McElreath, R., with Simpson, D., Margossian, C. C.,
  Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., Modrák, M., and Leos Barajas,
  V. 2026. _Bayesian Workflow_. CRC Press, corrected edition 20 July 2026. Cited
  by chapter and page: §2.2 (p. 20, what can and cannot be automated), §6.3 (p.
  109, what a single simulated-data fit shows), §8.2 (pp. 142–143, test
  summaries; no universal failure rule), §8.3 (p. 147, in-sample optimism of
  posterior predictive checks), §11.2 (pp. 195–196, initial values; approximate
  draws as starts), §11.4 (p. 198, four independent chains; R̂), §12.1 (p. 209,
  fit fast, fail fast), §12.3 (p. 218, many starts to find modes, then fewer
  chains from them), §12.5 (p. 234, refining an initialization scheme),
  §14.1–14.3 (pp. 249–253, SBC; the weak-prior clash; data-only rejection; few
  simulations beat none; posterior SBC), §15.4 (p. 258, scripts; Snakemake and
  `targets`), §30.5 (p. 469, dispersed initial points for reliable diagnostics).
- Gelman, A., and Rubin, D. B. 1992. "Inference from iterative simulation using
  multiple sequences." _Statistical Science_ 7(4): 457–472. The origin of R̂ and
  of the requirement that chains start dispersed; cited by the book at §11.4.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. 2021.
  "Rank-normalization, folding, and localization: an improved R̂ for assessing
  convergence of MCMC." _Bayesian Analysis_ 16(2): 667–718. The R̂ camdl reports.
- Säilynoja, T., Bürkner, P.-C., and Vehtari, A. 2022. "Graphical test for
  discrete uniformity and its applications in goodness-of-fit evaluation and
  multiple sample comparison." _Statistics and Computing_ 32, 32. The `γ`
  statistic `fit calibration` will report.
- Modrák, M., Moon, A. H., Kim, S., Bürkner, P.-C., Huurre, N., Faltejsková, K.,
  Gelman, A., and Vehtari, A. 2025. "Simulation-based calibration checking for
  Bayesian computation: the choice of test quantities shapes sensitivity."
  _Bayesian Analysis_. The SBC variant the book follows.
- Säilynoja, T., Schmitt, M., et al. 2026. Posterior SBC; cited by the book at
  §14.3. Retrieve the final reference before the follow-up proposal cites it.
- Yao, Y., Vehtari, A., and Gelman, A. 2022. "Stacking for non-mixing Bayesian
  computations: the curse and blessing of multimodal posteriors." _Journal of
  Machine Learning Research_ 23. Follow-up only.
- In-repository:
  `docs/dev/proposals/archive/post-alpha/2026-05-25-cli-init-and-params-ux.md`
  (the `init` / `init_mle` split this proposal collapses);
  `docs/dev/proposals/archive/pre-alpha/2026-04-17-synthetic-fit-replicates.md`
  (`[synthetic]` and the unbuilt `coverage.tsv`);
  `docs/dev/proposals/archive/pre-alpha/2026-04-19-refine-gates-scout-convergence.md`
  (the two gates, superseded here);
  `docs/dev/proposals/2026-08-29-honest-predictive-evaluation.md` (holdout and
  the `held_out` label);
  `docs/dev/proposals/2026-08-23-run-identity-and-store-contract.md` (the
  include-by-default identity rule this proposal's re-keys follow); gh#711,
  gh#831, gh#829, gh#830, gh#154, gh#369, gh#593, gh#871–873.
