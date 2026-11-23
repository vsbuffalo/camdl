# The `fit.toml` reference

A `fit.toml` is the runtime config for `camdl fit run` — it names the model, the
data, what to estimate, what to fix, and the one inference method to fit with.
It is **not** part of the model language: the `.camdl` file declares parameter
names, bounds, and priors; the `fit.toml` selects which parameters to estimate
and how. Every field below is verified against the parser (`config_v2.rs`).

For the workflow these configs drive, see `camdl docs workflow`.

## A complete example

```toml
store = "camdl-store"             # the run store's root (optional; relative
                                  # to this file — see "How paths resolve").

[model]
camdl = "model.camdl"

[data.observations]               # one key per observation stream in the model
cases = "data/cases.tsv"

[estimate]                        # the parameters to infer
beta  = { bounds = [0.001, 0.5], start = 0.04,
          prior = { log_normal = { mu = -2.0, sigma = 1.0 } } }
gamma = { bounds = [0.01,  1.0], start = 0.12,
          prior = { log_normal = { mu = -1.2, sigma = 0.5 } } }
s0    = { bounds = [0.01, 0.30] }  # initial state

[fixed]                           # held at a value, not estimated
rho = 0.6
k   = 10.0

[method]                          # the one way this file fits the problem
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 4
particles = 600
sweeps    = 300
# starts = "from_prior"           # the default here: every parameter has a prior
```

The file has two halves. Everything above `[method]` is the **problem** — the
model, the data, the estimate/fixed partition, the priors — and every reader of
a fit config reads it: `simulate --draws prior --fit`, `pfilter --fit`,
`profile --fit`. A file with no `[method]` at all is a complete problem for
those. `[method]` is the **inference**: how `fit run` fits that problem. A
second way of fitting the same problem is a second file with the same problem
half and a different `[method]` (`camdl fit new --from fit.toml
fit-if2.toml`
copies the file with provenance; edit the table). The run store puts both runs
in one problem folder, `camdl-store/<build>/fit/<model>-<hash>/`, because the
problem's identity leaves the method out.

## Sections

**`[model]`** — `camdl = "path/to/model.camdl"`.

**`[data.observations]`** — one key per observation stream declared in the
model, each mapped to a TSV path.

`[data.holdout]` (same shape) and `holdout_after = <time>` (under `[data]`) are
**applied at fit load** (gh#585). Under `holdout_after` a row trains if the
_end_ of what it measures is at or before the cutoff. The cutoff is a model-time
number, a date under a calendar-anchored model (the instant that day begins), or
an anchored form such as `last_obs - 6 weeks`, where `last_obs` is the latest
end of an observed row over the full files. For a `closing_at` or window-column
stream the end is the row's label, so the row labelled with the cutoff trains.
For a stream labelled by the start of its period (`covers = day(time)`,
`starting_on`, `ending_on`) the end is later than the label: under
`holdout_after = 10` the `day(time)` row labelled 10 covers day 10, ends at 11
and is held out. Every bound stream must keep a training row with a value, and
some row with a value must be held out: an `NA` row keeps its place among the
rows but is no observation to train on or to score. Under `[data.holdout]` every
row of the training files trains, and every row of the holdout files must end
strictly after the training extent, the latest end among the training rows of
every stream (tail-only).

Every fit records which rows it trained on in `fit.meta.json` as
`training_partition`: per stream the number of leading rows that trained and
their row digest, and the training extent. The fit prints the rows trained, the
extent and the first held-out row of each stream. A stored fit keeps the
held-out rows' dates and not their values, so `camdl compare` refuses to score
the held-out rows of a stored fit run, and `--in-sample` compares fits on their
training rows. To score a held-out window by hand, score the **full** series
with `camdl pfilter --save-prequential --score-from <extent>`, summing the
`log_score` of the `{STEM}.tsv` rows (the joint score per step). Scoring the
full series matters: scoring the held-out file on its own starts the filter from
the prior with nothing assimilated from the training window, which is a
different (and unfairly harsh) quantity. The recipe is spelled out under
"Out-of-sample validation" in `camdl docs inference`.

`camdl profile --fit` applies the holdout as `fit run` does, through the same
function, whether its data come from `--data` or from the config: under
`holdout_after` every cell fits only the rows that end at or before the cutoff,
and a `[data.holdout]` config's training files are fitted whole, its holdout
files checked against the bound training data by the same tail-only rule.

`[synthetic]` may stand beside `[data]`: `fit run` then fits the real data, and
the synthetic block records the truth a recovery check reads.

**`[data.tables]`** — one key per external input the model declares, each mapped
to the file that supplies it. A model declares an external input as a table,
`C : patch × patch = external("contact")`, or as a forcing's data,
`data = external("school")` (language spec §6.5 and §7). Each needs a file when
the model is run, and `[data.tables]` is where a fit config gives it:

```toml
[data.tables]
school  = "../data/build/calendar/school_c1.tsv"
contact = "../data/build/contact.tsv"
```

Every command that reads a fit config binds these files: `fit run`,
`pfilter --fit`, `profile --fit`, and through a stored fit `camdl predict` and
`simulate --draws posterior --fit`. A `--table NAME=FILE` on the command line
replaces the config's file for that name, as `--data` replaces a
`[data.observations]` entry. A run is refused, naming the input, when the model
declares an external input that nothing binds, when `[data.tables]` or `--table`
names one the model does not declare, or when the file does not hold what the
declaration reads: a forcing's file without its `time_col`, `value_col` or
`key_col`, or with no row for one of its levels; a table's file without exactly
one number per cell.

A bound file enters the fit's identity by its bytes, not its path: the same
bytes under another path are the same fit (and are reused), and one changed byte
is another fit. The fit folder records each file's SHA-256 in `fit.meta.json`
(`table_hashes`) and keeps a copy of its bytes in `inputs/NAME.EXT`. What a fit
replays (`camdl predict`, `simulate --fit`) is the model as it was bound, so
editing the source file after the fit cannot change a past fit's predictions;
`--table` on a replay is how to run it under another file on purpose. A config
with `[synthetic]` and no `[data]` has nowhere to write `[data.tables]`, so its
model cannot declare an external input.

**`[simulate]`** — the simulated window, as this config sets it for the model.
Its one key, `from`, is where the run starts, and overrides the model's
`simulate { from }`:

```toml
[simulate]
from = "first_obs"            # or -2, or "1950-01-01" under a model origin
```

The value is a model-time number, a date under a model `origin`, or an
observation anchor with a bare-word offset (`"first_obs - 1 week"`), the grammar
of `simulate --to`. `first_obs` is the start of the earliest observed period of
the training rows this config binds, so for weekly counts with
`covers = closing_at(time, 7 'days)` whose first row is labelled day 5 it is day
−2, the day that week opened. Every command that reads the config applies it:
`fit run`, `pfilter --fit`, `profile --fit` and `simulate --fit`, where
`simulate --from` takes precedence over it. The start moves with every start the
model derives from its own (the output and emit schedules, a recurring
intervention with no `from`), and a model whose literal `from` is also a
recurring schedule's start is refused rather than guessed (the language spec,
§18).

The start enters the fit's identity through the model, not through this key:
camdl writes the resolved start into the model before keying the fit, so
`from = "first_obs"`, `from = -2` and a model written with `from = -2 'days` are
one fit when they resolve to the same day, and two windows are two fits. The
stored fit records the model with its start, so `camdl predict` and
`simulate --fit <run dir>` replay it.

**`[estimate]`** — the parameters to infer. Each value is an inline table:

| key                         | meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bounds = [lo, hi]`         | search range. _Optional_ — defaults to the model's `parameters { p : rate in [lo,hi] }` range; a `fit.toml` `bounds` may only **narrow** it, never loosen.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `start = X`                 | the base starting value. Optional — defaults to a draw within the bounds (a model carries no parameter value to start from). See "Where a chain starts" below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `prior = { … }`             | prior distribution. **Required** for a `pgas`/`pmmh`/`mh` method. It also picks the default chain starts under every method, `if2` included: `from_prior` when every estimated parameter has one. See "Priors" below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `perturb_only_at_t0 = true` | an initial-state parameter (e.g. `s0`, `i0`) — perturbed at t=0 only, never at an observation. It is IF2's perturbation schedule, so only `if2` accepts it: any other method refuses the config at load rather than run without it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `rw_sd = X`                 | IF2 per-parameter random-walk SD, on the natural scale, converted once to the parameter's transformed coordinate (at the box's midpoint, or at the start on a half-bounded domain); IF2 then takes that same step in z for every chain and particle. Optional — the default is a step in z per transform: 1/20 on a box finite at both ends, 1/5 on a half-bounded domain such as `[0, ∞)`; these are starting heuristics chosen for robustness to poor starts, not optimal values (an unbounded `real` or `instant` has none and needs one). `pmmh` and `mh` also read it: five times the converted step is the initial proposal SD; with no `rw_sd` they start from 0.25 in z (the adaptive proposal takes over from there). `pgas` does not read it: its MH-within-Gibbs θ-step (`--no-nuts`) starts from 0.25 in z, and NUTS, its default, has no step to set. The `nl-*` optimizers ignore it. So `rw_sd` is read by `if2`, `pmmh` and `mh` only. |

`camdl profile --fit` reads `[estimate]` the same way: without `--rw-sd`, each
cell estimates `[estimate]` less the swept and pinned parameters, and an IF2
cell steps each by its `rw_sd` or the default above. A swept parameter is listed
here too (its bounds bound the grid); each cell pins it.

**`[fixed]`** — `param = value` for every model parameter you are _not_
estimating. camdl requires every declared parameter to be either estimated or
fixed.

**`[method]`** — the one inference method. `algorithm` picks the method and
`backend` the simulator it fits against: `chain_binomial` for the
stochastic-process methods (`if2`, `pgas`, `pmmh`, `pfilter` — the ones that
need chain-binomial process noise and `balance`), or `ode` for the
deterministic-likelihood methods (`nl-sbplx` / `nl-bobyqa` / `nl-lbfgs` MLE and
the Bayesian `mh`). Each algorithm is valid on exactly one backend —
`camdl fit methods` lists the pairs, and an invalid pair is rejected at load.
`starts` says where the chains begin (below). There is no map of named stages
and no execution order: a file declares one method, and a pipeline is two files
run in turn.

**`[config]`** — fit-wide simulator settings: `dt`, the integrator step for a
model that declares none (default `1.0`). A model that declares
`simulate { dt = … }` is integrated at that step by every method; a
`[config] dt` that differs from it is refused, and an agreeing one is accepted.
A `dt` written at the top level of the file is a typo, not a setting. (The
forward backend for synthetic-data generation is `[synthetic].backend`, not a
`[config]` setting — gh#241; the method declares its own `backend`.)

`allow_forcing_extrapolation = true` lets the fit evaluate a forcing outside the
range it is defined on — an `interpolated` forcing before its first or after its
last data point, a `bspline` outside its window — holding its end value there.
Without it such a fit is refused, naming the forcing, its range and the side it
was evaluated on. Under it each such forcing gets one warning naming its range
and the span held, and the fit's `run.json` records the override. The setting is
part of the fit's identity (a fit made under it and one refused without it are
different computations), and `fit run
--allow-forcing-extrapolation` sets it.
`camdl predict` of such a fit runs under it too.

Observation times need not lie on the grid `t_start + k·dt`. `if2`, `pfilter`
and `pmmh` step exactly to every observed period's start and end; `pgas` steps
only on its grid and refuses a run in which any period starts or ends off it,
naming the row and the two grid times either side — choose a `dt` that puts
every boundary on the grid, or fit with `if2` or `pmmh`. There is no setting for
this: `[config] obs_alignment` is removed, and a file that carries it is
refused.

## One model under several covariate files

A model of measles in a school-age population scales transmission by a daily "in
term" indicator, read from whichever school calendar the run is given:

```camdl
forcing {
  term : interpolated 'ratio {
    data      = external("school")
    time_col  = "date"
    value_col = "in_term"
    method    = constant
  }
}
```

Two fit configs differ only in the calendar they bind:

```toml
# fit_c1.toml
[model]
camdl = "measles.camdl"

[data.observations]
cases = "../data/build/cases.tsv"

[data.tables]
school = "../data/build/calendar/school_c1.tsv"

# … [estimate], [fixed] and [method] as for any fit
```

```toml
# fit_c2.toml: the same file, except
[data.tables]
school = "../data/build/calendar/school_c2.tsv"
```

```bash
camdl fit run fit_c1.toml
camdl fit run fit_c2.toml
```

There is one model file, so `camdl compare` and the store see one model fitted
twice: the two fits differ in their `table_hashes`, their run ids and their
likelihoods, and in nothing that came from the model. A third calendar is a
third config or, for a quick likelihood at fixed parameters, a flag:

```bash
camdl pfilter measles.camdl --fit fit_c1.toml --params theta.toml \
  --table school=../data/build/calendar/school_c3.tsv
```

## One model over several data windows

A model whose run starts one week before its data begin can say so once, and
leave the window to the data:

```camdl
simulate { from = first_obs  to = 5479 'days  dt = 1 'days }
```

Each data window is then a config. Two configs that bind the 1950–1963 and the
1950–1960 notifications fit the same model from two starts, each at the opening
of its own first observed week:

```toml
# fit_full.toml
[model]
camdl = "he2010.camdl"

[data.observations]
cases = "../data/build/cases_1950_1963.tsv"
```

```toml
# fit_refit.toml: the same file, except
[data.observations]
cases = "../data/build/cases_1950_1960.tsv"
```

When the data do not fix the start (scoring a held-out window from a model
fitted on another), the config states it:

```toml
[simulate]
from = "1961-01-01"   # a date needs `origin = date(...)` in the model
```

## How paths resolve

**Every path written in the `fit.toml` is relative to the `fit.toml` itself** —
`[model].camdl`, `[data].file`, each `[data.observations]`, `[data.holdout]` and
`[data.tables]` stream, `store`, and a `starts` source that is a file
(`from_params = "theta.toml"`). Absolute paths pass through unchanged (and draw
a portability warning, since they pin the config to one machine's layout).

The rule is the one Cargo and `pyproject.toml` use: a path written **in a file**
anchors at that file, so the config is relocatable as a unit and runs the same
from any working directory. What does _not_ come from the file — the
`camdl-store/` default when no `store` is declared, and `CAMDL_STORE` — anchors
at the working directory instead, since that is the frame you typed it in.

Concretely, for this layout:

```
<repo>/camdl/fit.toml
<repo>/camdl/bvd_province.camdl
<repo>/data/build/camdl/cases.tsv
```

every path takes the same base, whichever directory you run from:

```toml
store = "../camdl-store" # <repo>/camdl-store

[model]
camdl = "bvd_province.camdl" # <repo>/camdl/bvd_province.camdl

[data.observations]
cases = "../data/build/camdl/cases.tsv" # <repo>/data/build/camdl/cases.tsv
```

`fit run` prints the resolved output location as an **absolute** path at start,
so you can confirm where the run tree is going before it is written.

## Where a chain starts

`starts`, under `[method]`, is one key for one concept: where the chains begin.
Its value is a bare rule name, or a one-key table naming a rule and its source.

```toml
[method]
starts = "from_prior" # one independent draw per chain
starts = "uniform_unconstrained"
starts = "lhs"
starts = "uniform"
starts = "single" # every chain at the base point
starts = { from_posterior = "@base" } # one row of @base's posterior per chain
starts = { from_posterior = "@base", chains = [0, 1, 3] } # rows of those chains only
starts = { from_mle = "@mle" } # every chain at @mle's point estimate
starts = { from_params = "theta.toml" } # every chain at the values in the file
```

The rules fall into two kinds, and the kind is what R̂ needs to know.

**Spread** rules give each chain its own start, so the between-chain R̂ can say
whether the chains found the same posterior:

| `starts`                          | where the chains start                                                                                                                                      |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `from_prior`                      | one draw from the declared priors per chain, restricted to `bounds`; base point unused                                                                      |
| `uniform_unconstrained`           | Stan-style boundary-avoiding draws on the unconstrained scale; base point unused                                                                            |
| `lhs`                             | Latin-hypercube stratified over `bounds`; base point unused                                                                                                 |
| `uniform`                         | chain 0 at the base point; the rest uniform within `bounds`                                                                                                 |
| `{ from_posterior = "<handle>" }` | a distinct row of a stored fit's posterior per chain (or of a draws TSV named directly); a source that lacks a column for an estimated parameter is refused |

`from_posterior` assigns rows without replacement, in an order set by the seed,
so no two chains start from the same row while the source has at least as many
rows as chains. With fewer rows, every row is used before any is reused, and a
warning names how many chains share a start.

**Point** rules put every chain at one point. R̂ is then _not assessed_ — chains
that began together agree by construction, and the summary says so instead of
reporting a pass:

| `starts`                     | where the chains start                                                                                                                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `single`                     | every chain at the base point                                                                                                                                                                                                        |
| `{ from_mle = "<handle>" }`  | every chain at a stored optimizer fit's point estimate (`if2`, `nl-*`; a `pgas`/`pmmh`/`mh` source is refused, since a sampler writes no MLE: use `from_posterior`); a parameter absent from the source falls back to bounds-uniform |
| `{ from_params = "<toml>" }` | every chain at the values in a flat params TOML                                                                                                                                                                                      |

A **handle** names one stored fit run: `@label`, a prefix of the run's `run_id`,
or the run's folder. A `fit.toml` never names a run and is refused, with the
runs it made listed. A source whose stored verdict is not converged is refused,
since starting from an unconverged fit launders its multi-modality into this
one; `fit run --allow-nonconverged-source` records the choice to start from it
anyway. A `@label` names the latest invocation of `fit run --label` with that
label whose runs all completed, and `@label~1` the one before it
(`camdl-run-spec.md` §6.8). The run the handle resolved to is recorded by its
`run_id` beside the handle as typed, and that run id is what the new run's
identity keys on, so moving the label later does not change what this run
started from.

**Starting from some of a source's chains.** When a source run is multimodal —
some chains settled in a mode with far lower posterior density than the others —
the next run should start from the chains that agree. `chains = [..]` restricts
`from_posterior` to those chains' rows of the source's `draws.tsv`, and
`--starts-chains 0,1,3` does the same on the command line, on `fit run` and on
`camdl profile` alike. The whole run's verdict is then replaced by the selected
chains' own R̂: the source passes when they agree (R̂ below 1.05 for every
parameter column that varies), and is refused otherwise, or when one chain
leaves R̂ uncomputable; `--allow-nonconverged-source` lifts that refusal. An
empty list, a chain id given twice, an empty entry (`0,,1`) and an entry that is
not a whole number are refused, as they are for `--exclude-chains`. A chain the
source does not hold, or a draws file with no `chain` column, is refused. The
subset is part of the run's identity, so two subsets of one source are two runs.
Compare the modes' log-posterior densities before choosing: dropping chains from
a mode the posterior genuinely weights would remove part of the posterior.

**A `pfilter` method scores one θ**, so the file must determine it: a `start`
for every estimated parameter (then `starts` defaults to `single`), a sourced
rule (`from_mle`, `from_params`, `from_posterior`), or `starts = "from_prior"`
asked for by name for one prior draw. Anything else — including priors alone, or
a bounds-based spread rule — is refused rather than scored at a random draw.

**The default.** When the file does not say, `starts` is `from_prior` if every
estimated parameter has a prior the chains can be drawn from, and
`uniform_unconstrained` otherwise; `fit run` prints which, and why. A
bounds-uniform draw at province scale is routinely a start the filter cannot
score, because a fixed relative error in a rate is a standardised residual that
grows with the square root of the population; a prior carries the scale.

**`[estimate].start` sets the base point** — one θ. It is load-bearing under
`single` and `uniform` (and for a parameter a `from_mle` / `from_params` source
does not name); the other rules ignore it on purpose, and `fit run` notes when a
declared `start` was unused. The three spreading rules — `uniform`, `lhs`,
`uniform_unconstrained` — fall back to the base point at `chains = 1`, since
with one chain there is nothing to spread. The source-reading rules do not: they
read from their source however many chains you asked for.

`starts` is part of the method's identity: the same method under a different
rule is a different run, and a sourced rule folds the source file's content into
the run's dependencies, so a regenerated upstream re-keys the fits that started
from it. `fit run --starts <spec>` overrides the file with the same spellings
(`--starts from_prior`, `--starts from_mle=@mle`).

Every multi-chain sampler (`if2`, `pgas`, `pmmh`, `mh`) writes a
`chain_starts.tsv` recording where every chain actually began, before any
perturbation, and names the rule that supplied those values. That file, not the
config, is the authority on what a run did — a fit started with `from_mle`
records `from_mle` on every row, because every chain starts at the upstream
fit's single point. The optimizer-only methods (`nl-sbplx`, `nl-bobyqa`) write
none.

A spread rule is a lottery over one draw, so a start the filter cannot score is
one unlucky draw rather than a verdict on the chain: it is redrawn under the
same rule, up to ten times per chain, and the chain is refused only when every
attempt fails. Every attempt is in `chain_starts.tsv` — one row per attempt,
with `attempt`, `status` (`accepted`, `rejected`, or `refused`), the filter's
`ess` at refusal and the `reason` — and the header line counts the redraws
(`retried=K`). A point rule has nothing to redraw and gets its one try.

> **Unknown keys are rejected.** A misplaced or misspelled key is a hard error
> naming the offending key — `fit.toml` is parsed strictly. A top-level `dt` (it
> belongs in `[config]`) or `particle` (it is `particles`, under `[method]`)
> fails at load rather than being silently dropped, so a sweep that varies a
> typo'd knob can't quietly produce identical fits.

## Migrating a `[stages]` file

A file that still carries the stage map is refused at load, and the message
spells the rewrite for that file:

```
legacy table `[stages.posterior]`
  replacement: rename to `[method]` and run it with
    camdl fit run fit.toml
  a file carries one `[method]`; put `[stages.scout]` in its own file
  `init_mle = "scout"` has no replacement in the file: it started every chain at
  scout's point estimate, which makes R̂ uninformative. Run scout first and, if a
  warm start is wanted, write one of
    starts = { from_posterior = "@scout" }   # one draw per chain (keeps R̂ meaningful)
    starts = { from_mle = "@scout" }         # every chain at one point (R̂ not assessed)
  `init = "lhs"` becomes `starts = "lhs"`
  See `camdl docs fit-toml`.
```

Nothing is converted silently. `init` is `starts`; `init_mle` is one of the two
sourced rules, chosen by the author, with the upstream run first and named by
handle; `survey_top_k` with `survey_path` / `survey_top_k_n` is refused, and
`from_prior`, or `from_posterior` from a short run, takes its place; the
top-level `fit_starts` key, which no runner read, is gone with it (`from_prior`
is now the default whenever every estimated parameter declares a prior).

## Where scoring begins

An **incidence** observation (a weekly case count, say) is the flow accumulated
over the period its row covers, and the model states that period on the stream —
`covers = day(time)`, `starting_on(...)`, `ending_on(...)`, `closing_at(...)`,
or `window_start`/`window_stop` columns (`camdl docs data`, "What a row
covers"). That declaration also decides where the likelihood begins. The first
row's period opens where the declaration says, so a model whose dynamics start
well before the data — `simulate.from` in 2011 so births and SIA/MCV covariates
shape the susceptible pool, case data from 2014 — simulates the 2011–2014 span
with the full stochastic dynamics and scores none of it. The first datum is
scored against its own declared period, never against the whole warm-up, and
there is no `fit.toml` setting for that boundary. A first period that opens
before `simulate.from` is an error naming both times: that span is never
simulated.

`fit.toml` used to carry a `condition_from` key for this. It is removed, and a
leftover key — top-level, or under `[data]` — is a hard error rather than an
unknown-field rejection:

```
condition_from = ... is no longer a fit.toml key: an incidence stream now states
what each row covers (`covers = ...` or window columns in the model), and a
declared first period opens where it says, so the warm-up before the first row
is discarded without a separate setting. Delete the key.
```

`camdl pfilter`, `camdl profile` and `camdl predict` bind the same declared
periods, so a fixed-θ log-likelihood or a predictive row covers exactly the
window the fit scored. The history is in `camdl docs language-changes`.

## Priors

Externally-tagged inline tables (the wire format matches the IR emission):

```toml
prior = { log_normal = { mu = 0.0, sigma = 1.0 } }
prior = { normal = { mean = 0.0, sd = 1.0 } }
prior = { beta = { alpha = 2.0, beta = 5.0 } }
prior = { log_uniform = { lower = 1e-5, upper = 1e-2 } } # uniform on the log scale
prior = { truncated_normal = { mean = 0.7, sd = 0.2, lower = 0.3, upper = 1.0 } }
prior = { uniform = { lower = 0.0, upper = 1.0 } } # explicit bounds
prior = { uniform = {} } # uniform over the param's `bounds`
prior = { half_normal = { sigma = 1.0 } }
prior = { flat = {} } # explicit "flat on purpose" — only valid in fit.toml
```

The empty `uniform = {}` form is uniform over the parameter's `bounds` (the
`[estimate.<name>].bounds`, falling back to the model's `in [lo, hi]`) — a
convenience so you don't repeat the interval. It requires bounds from one of
those sources. `truncated_normal`'s `lower`/`upper` must equal the parameter's
bounds (the prior's support and the search box are the same interval).

**Precedence:** a `fit.toml` `[estimate].prior` overrides the model's `~`
declaration, with a warning naming both (see
[How `fit.toml` relates to the model](#how-fittoml-relates-to-the-model)). If
neither is present, `fit run` with a Bayesian method **refuses to start**,
naming each parameter with no prior — camdl never falls back to a flat prior
silently, because the prior shows up in the posterior. (`profile`'s per-cell
PMMH is the one exception: it falls back to flat with a warning.) The explicit
`{ flat = {} }` is how you say "flat here, on purpose". A flat prior is not a
distribution the chains can be drawn from, so a parameter left flat also moves
the default `starts` from `from_prior` to `uniform_unconstrained`.

## Method algorithms

| `algorithm`              | backend          | role                                              | key fields                                                                            |
| ------------------------ | ---------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `if2`                    | `chain_binomial` | iterated filtering → MLE                          | `chains`, `particles`, `iterations`, `cooling` (+ `cooling_target_iters`, default 50) |
| `pgas`                   | `chain_binomial` | particle Gibbs + NUTS → posterior                 | `chains`, `particles`, `sweeps` (+ `burn_in`, `thin`, `max_tree_depth`)               |
| `pmmh`                   | `chain_binomial` | particle marginal MH → posterior                  | `chains`, `particles`, `iterations`                                                   |
| `pfilter`                | `chain_binomial` | particle filter at fixed θ → log-likelihood + ESS | `particles`, `replicates`                                                             |
| `nl-sbplx` / `nl-bobyqa` | `ode`            | NLopt deterministic optimizer → MLE               | `chains` (LHS starts) (+ `max_evals`, `tolerance`)                                    |
| `mh`                     | `ode`            | MH on the deterministic ODE marginal → posterior  | `chains`, `iterations` (+ `burn_in`, `thin`, `adapt`, `adapt_start`, `burnin_dt`)     |
| `nl-lbfgs`               | `ode`            | NLopt L-BFGS (forward-sensitivity gradient) → MLE | `chains` (LHS starts) (+ `max_evals`, `tolerance`)                                    |

The `ode`-backend methods fit the **deterministic marginal likelihood**
`p(y | θ, ODE skeleton)` rather than the stochastic `p(y | θ)` — a different
statistical object, appropriate for equilibrium or large-population models. `mh`
is the Bayesian sampler there and is gradient-free. `nl-lbfgs` requires a
differentiable model (the capability check refuses an undifferentiable gradient,
an adaptive `rk45` integrator, a scheduled effect whose jump the sensitivity
cannot cross — a reactive or estimated-time one, a `transfer(count = …)`, a
state-dependent amount — or an initial condition the gradient path cannot seed;
fixed-time fraction transfers, `add` and `set` effects are carried); `nl-sbplx`
and `nl-bobyqa` carry no such requirement. See `camdl docs inference` (the
ODE-backend fitting section) for when to pick which.

A `[method]` key the method's own settings leave with nothing to do is refused
at load, not ignored: `max_tree_depth` or `dense_mass` under `use_nuts = false`
(the θ-move is then Metropolis-within-Gibbs), and `adapt_start` under
`adapt = false`.

```toml
# A Bayesian fit on the ODE skeleton.
[method]
algorithm = "mh"
backend = "ode"
chains = 4
iterations = 5000
burn_in = 1000
```

Common to every method: `starts` (above), and the `dt_check` sub-table where the
method runs a dt-convergence audit.

### Coarse warm-up step (`burnin_dt`, `mh` on `ode`)

A model often starts well before its first observation so that the transient
settles — years of warm-up before a few months of data. Nothing in that span is
scored, yet by default it is integrated at the fine `dt`, and for a long warm-up
it dominates the cost of each likelihood evaluation. `burnin_dt` integrates the
unscored span `[t_start, first observation)` with a larger fixed RK4 step and
switches back to `dt` at the first observation:

```toml
[config]
dt = 1.0

[method]
algorithm = "mh"
backend = "ode"
chains = 4
iterations = 20000
burnin_dt = 7.0 # coarse RK4 step on the unscored warm-up only
```

On the Garki `ctl_bb` mosquito model (`mh`, `dt = 1`) `burnin_dt = 7` cut the
per-evaluation cost 3.5× with a negligible change in log-likelihood. The coarse
step does change the warm-up trajectory, so check that the fitted posterior
agrees with a run at `burnin_dt` unset before relying on it.

It is off by default; `burnin_dt = dt` is also off. A fit is refused, naming
`burnin_dt`, when:

- the fit has an **incidence** stream — the flow accumulated into the first
  scored period is computed from the warm-up, so coarsening could bias a scored
  datum. Only prevalence (state-at-a-time) streams are supported;
- `burnin_dt < dt` — a coarse step must be larger than `dt`;
- the first observation is at or before `t_start` — there is no warm-up to
  coarsen;
- the model has a `balance {}` or `events {}` block — both act every substep, so
  a coarse step would change what they do;
- the integrator is `rk45` — an adaptive step has no fixed size to coarsen; use
  `rk4`;
- a recurring intervention would fire twice within one coarse step.

`burnin_dt` is part of the fit's identity: two runs that differ only in it get
distinct run ids and never share a cached posterior. The other methods do not
take it — the particle methods (`if2`, `pgas`, `pmmh`) simulate the warm-up
stochastically, and the NLopt optimizers integrate at `dt` throughout.

## How `fit.toml` relates to the model

The model file is the source of truth for what _can_ be estimated; the
`fit.toml` chooses and configures. Specifically:

- **Bounds** default to the model's declared range; `fit.toml` can only narrow.
- **Priors** default to the model's `~` declarations; `fit.toml` overrides. An
  override is never silent: when a parameter has both, `fit run`, `profile` and
  `simulate --draws prior` print one warning before any work, naming the
  parameter, both priors, and that the `fit.toml` one is used. A Bayesian fit
  also records the displaced model prior in `fit.meta.json`
  (`resolved_priors[].overridden_model_prior`). If the override is not
  deliberate, delete the `prior`; if it should hold for every fit, move it into
  the model.
- **Transforms** follow from the parameter's declared type and its bounds
  (`docs/camdl-inference-spec.md` §3.2), and there is no override: the transform
  maps the real line onto the bounds, so the parameter stays inside them and no
  target has a wall at a bound. A prior whose support is narrower than the
  bounds is refused (`log_normal` on bounds below zero, `beta` outside
  `[0, 1]`).
- **Bounds stay inside the kind.** Where the model declares no `in [lo, hi]`,
  `bounds` may narrow the kind's support (`[0, ∞)` for `rate`, `positive`,
  `count`, `duration`; `[0, 1]` for `probability`) but not leave it; a signed
  range is declared in the model (`phi : duration in [-120, 120]`).
- **Unbounded parameters.** With no `bounds` here and no `in [lo, hi]` in the
  model, a `real` or `instant` parameter is searched over `(−∞, ∞)` with the
  identity transform; the positive kinds (`rate`, `positive`, `count`,
  `duration`) over `[0, ∞)` with θ = eᶻ; a `probability` over `[0, 1]`. A `real`
  or `instant` on `(−∞, ∞)` has no automatic `rw_sd` (there is no scale to take
  one from); give `rw_sd`. At two or more chains the `uniform`, `lhs` and
  `uniform_unconstrained` starts refuse it, the default `uniform_unconstrained`
  included. `from_prior` draws it from its prior, and `single` starts every
  chain at its `start`.
- **Hierarchical parameters** (`a ~ normal(mu, 1)`) may not have bounds that cut
  into their prior's support under `pmmh` or `mh`: the prior is not renormalised
  to the bounds, and for a hierarchical parameter the missing constant depends
  on the hyperparameter, which would bias it. Leave the bounds off (a `rate`
  keeps `[0, ∞)`, the whole support of a `log_normal`), or remove the
  hierarchical prior.

So the minimal `fit.toml` for a model that already declares bounds and priors is
just `[model]`, `[data]`, an `[estimate]` listing names (no per-param fields),
`[fixed]`, and a `[method]`.
