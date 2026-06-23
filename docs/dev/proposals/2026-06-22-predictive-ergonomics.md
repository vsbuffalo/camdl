# Predictive-output ergonomics: a convergence-gated `predict` / `latent` verb + tidy CAS artifact

Status: draft Relates: #273 (fixed-param backfill — execution step 1), #279
(obs/dimension schema + tidy postpred artifact), #277 (model-criticism metrics +
held-out eval), #276 / #267 (latent-trajectory consolidation), gh#269
(per-stream prequential — the one-step source), gh#86 (`--draws prior` — landed)
Supersedes: the "unify the two predictive objects under one `kind` key" comment
on #279

## Background: predicted-vs-observed, and the three predictive objects

_For a reader new to the workflow; skip if you live in it._

A fit takes a model and the **observed data** — the numbers actually recorded
(say, the count of new Ebola onsets reported in district _Bo_ in week 7) — and
returns a **posterior**: a probability distribution over the model's unknown
parameters given that data.

The first thing anyone does with a posterior is ask: _does the model actually
reproduce the data?_ The universal way to answer is **predicted-vs-observed** —
overlay the observed points on the model's **predictive distribution**, the
range of counts the fitted model says it _would_ produce. Observed points inside
the predicted band mean the model is consistent with the data; points outside
mean it is not. One value over time, observed dots on a predicted ribbon, one
panel per place: this is the workhorse diagnostic, and it is the _same shape_
for every model.

Two terms recur:

- a **stream** is one observed quantity the model is fit to (`onset`, `deaths`);
  a model may have several.
- a stream is indexed by **dimensions** — here `patch` (district) — and each
  combination of levels is a **stratum** (district _Bo_). "Per-district onset"
  is the `onset` stream with `index_dims = [patch]`; a single national series is
  the same stream with no index dims.

"Predictive" is not one object. There are **three**, all the same shape — a
distribution per `(time, stratum)` — differing only in _which_ question they
answer:

1. **Free-forward posterior predictive** — _run the fitted model forward from
   the start; what data would it generate?_ The generative check: left to its
   own dynamics, does the model look like reality?
2. **One-step-ahead (prequential)** — _given everything observed up to last
   week, what is predicted for this week?_ The honest short-horizon forecast: it
   is told the real past at each step, a fairer test than the free-forward run.
3. **Posterior latent state** — not an observable at all, but the _hidden_
   quantity behind the data: the true infection incidence that the reported,
   under-counted cases are a noisy shadow of. Often the scientifically
   interesting output — the real epidemic, not the reported one.

Because all three are "a predictive distribution per `(time, stratum)`," one
artifact can carry all of them (§3) — and, today, an analyst rebuilds the same
machinery three times to get them.

## The shape of the problem

A camdl fit produces a posterior. Almost everything an analyst does _next_ —
predicted-vs-observed panels, posterior latent incidence, calibration checks —
is a _predictive_ object: a distribution over an observable (or a latent state)
per `(time, stratum)`. camdl owns every input to that object (the draws, the
fixed parameters, the observation model, the dimensions, the observation
cadence) but hands back none of it assembled. So the analyst reconstructs it by
hand, and the same reconstruction is rebuilt in every analysis script.

This proposal is grounded in field evidence: a 14-district spatial SEIRD built
in camdl (the camdl friction log, entries F11–F23), where the predictive
post-processing grew to **twelve scripts that each shell `camdl simulate` to
rebuild predictions**, plus a hand-rolled Python module whose only job is to
paper over the gaps. An audit of that code found three places where the
hand-reconstruction does not merely cost effort but **produces a statistically
wrong figure**. That is the real motivation: the goal is not "fewer lines," it
is _making the wrong figure unrepresentable_.

### The reconstruction, as one pipeline

Every predicted-vs-observed panel runs the same steps. Annotated with _what
camdl already knows but does not hand over_:

| Hand-written step (every script)                                                                                                        | What camdl already has                                     |
| --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| glob `…/<run>-<hash>/NN-posterior-<hash>/seed_*/chain_k/trace.tsv`, guard torn final lines, skip the empty resume-stub dir              | the run-store layout (CAS)                                 |
| filter out burn-in, concatenate chains, subsample                                                                                       | the declared `burn_in` (in the fit)                        |
| **re-inject the fixed parameters** (`sigma, gamma, ifr, iota, I0`) as constant columns, because the trace carries only estimated params | the `[fixed]` block (#273)                                 |
| write `draws.tsv`, clear a stale dir, shell `simulate --draws … --obs-only-dir`                                                         | the predictive engine (exists)                             |
| per stream: read `onset_<p>.tsv`, prefix-match the value column, pivot draw×time wide, float-sort the time headers                      | the stream list, dims, and value column (#279 schema; F12) |
| `quantile([5,25,50,75,95])` → ribbon                                                                                                    | a quantile reduction (trivial)                             |

The hand-rolled module's own docstring states the pattern outright: it exists
"as a bridge while camdl#86 is open." gh#86 has since landed, so that piece is
now _dead code that re-declares the model's priors in Python_ — a second source
of truth for distributions that live in the `.camdl`. The watcher tool is
blunter still, naming itself "the only module that knows the camdl run-store
layout … if camdl grows a sanctioned API later, swap this file and nothing
downstream changes." The consumer has already designed for camdl to absorb this.

### Three predictive objects, three hand-rolled pipelines

The reconstruction above is only the _first_ of three predictive objects an
analyst needs, each the same shape — a predictive distribution per
`(time, stratum)` — and each today reconstructed separately:

1. **Free-forward posterior predictive** `p(y_rep | y)` — launched from the
   origin (the generative check). The twelve `simulate`-shelling scripts.
2. **One-step-ahead prequential** `p(y_t | y_{1:t-1}, θ̂)` — conditions on the
   assimilated past (the honest forecast). A _separate_ artifact
   (`pfilter --save-prequential`, gh#269); on the spatial model this meant
   parsing a 35 MB per-step JSON.
3. **Posterior latent state** — the true incidence/prevalence behind the
   reported cases, from the _conditioned_ trajectories. No first-class extractor
   (F20); free-forward `simulate` at the draws **smears the median** across
   jittered stochastic take-offs, so it must come from the conditioned paths —
   which on the spatial run had directional conservation violations (F21), so
   they could not be trusted either.

## Why this is a correctness problem, not just ergonomics

The audit of the consumer code found three reconstructions that pass review and
yet are wrong. Each has the same shape: **camdl knows X, the analyst re-encoded
X by hand, and the re-encoding was wrong.**

1. **Predictive bands from non-stationary draws.** Two scripts render 5/50/95
   bands over draws explicitly filtered to a hand-picked, _pre-burn-in_ window
   (`SKIP=100`, comment: "pre-burn-in so this is provisional"), one of them also
   hand-dropping "the stuck chain 1." Quantiles over warm-up, hand-selected
   chains are not posterior quantiles — the band is biased and its spread is
   meaningless. The figures are title-labeled "PROVISIONAL," but they reuse the
   **same band code** as the converged scripts; the only thing separating a
   valid figure from an invalid one is a title string and an eyeballed integer.

2. **Burn-in as a per-script literal.** Four scripts hardcode their discard
   count (`250`, `250`, `100`, `60`) as bare integers, none read from the fit's
   own `burn_in`. This directly selects which draws enter the predictive, and
   drifts silently from what the fit actually used.

3. **Latent state at the posterior _median_ θ.** The "smoothed latent states"
   figure takes `median` of every parameter's marginal, builds one parameter
   vector, and runs the smoother at that single plug-in point. It carries
   smoothing noise but discards _all_ parameter uncertainty, and the
   median-of-marginals is not even a posterior point. It passes review because
   the paths wiggle — it looks like a posterior band. It is the F20 median-smear
   trap one level up.

A verb that (a) reads burn-in / fixed / strata _from the run_, (b) **refuses to
run on a non-converged fit**, and (c) integrates over θ rather than plugging in
the median makes all three unrepresentable. That is the design target.

## The duplicated-source-of-truth tax

Beyond the three bugs, the analysis re-encodes model facts as literals that
drift:

- **Priors** re-declared in Python (now dead, post-gh#86, but the pattern
  recurs).
- **Fixed parameters** re-typed in eight files with _disagreeing_ contents — a
  real hazard: a script that pins `iota` while the model _estimates_ it
  overwrites the posterior `iota` with a constant in the predictive.
- **The stratum (district) list** spelled out longhand in ~12 scripts, while two
  scripts correctly read it from the dimension's data file. The friction log
  already records the stratum set silently growing 6 → 14.

Each of these is metadata camdl has in the model and the fit config. Exposing it
(the #279 schema descriptor) lets both the verb and any remaining bespoke script
be generic instead of carrying a hand-typed copy.

## Design

Types and surfaces first; the CLI is a projection of these.

### 1. The run's observation/dimension schema (the metadata seam)

Per fit, emit a machine-readable descriptor of structure camdl already holds —
the streams, their indexing dimensions, the dimension levels, the value kind,
and the likelihood family. This is #279 Part 1, and it is the load-bearing
piece: it lets a consumer facet a stream by its `index_dims` and label panels by
level names _for any model, with no DSL parsing_, and it removes the duplicated
stratum lists. It must reuse the observation-system's internal stream/dimension
structures (`2026-06-06-observation-system.md`), not re-derive them.

```json
{
  "dimensions": { "patch": { "levels": ["Bo", "Bombali", "..."] } },
  "streams": [
    {
      "name": "onset",
      "time_column": "time",
      "index_dims": ["patch"],
      "value_column": "onset",
      "value_kind": "count",
      "likelihood": "neg_binomial"
    }
  ],
  "parameters": {
    "estimated": ["R0", "kappa", "rho", "..."],
    "fixed": { "sigma": 0.11, "gamma": 0.13, "ifr": 0.67 }
  }
}
```

It lives in the run's `fit.meta.json` — consumers like the watcher already read
that file, so the schema is one less artifact to discover. The descriptor marks
each parameter as **estimated or fixed** (the pinned values themselves live in
the fit's `[fixed]` block). That role marking is the seam a pairplot uses to
drop fixed parameters _by role_ — not by a fragile "which columns are constant"
scan, which would wrongly drop a near-degenerate estimated parameter. It is
load- bearing because the canonical `draws.tsv` _does_ carry the pinned
constants as constant trailing columns (so it re-simulates standalone), and a
pairplot reading it must know which of those columns to ignore.

### 2. `camdl fit predict <run>` — the verb that owns the reconstruction

One verb encapsulates the entire pipeline table above and writes a tidy
artifact. Its contract:

- **Resolves the posterior tail from the run.** Reads the post-warmup draws
  using the fit's _own_ `burn_in` (not a CLI literal), across chains, from the
  stable store path. This subsumes the trace-glob / torn-read / burn-in-literal
  steps — and is the missing `--draws posterior --fit` resolution (today
  `--draws` resolves only `uniform`, `prior`, and a file path;
  `main.rs:909,918`).
- **Fills missing parameters from the fit (#273).** The canonical `draws.tsv`
  already carries the pinned constants as constant trailing columns and
  re-simulates standalone; the raw per-chain `trace.tsv` carries only the
  estimated dimensions, so a predictive built from a trace (the mid-run look) is
  missing them. The verb resolves this exactly as #273's lead recommendation
  does: fill any parameter _absent_ from the draws source from `[fixed]`,
  **never overwriting a column that is present**. That fill-missing-only rule
  also closes the footgun the issue documents — a hand-rolled fixed-map silently
  clobbering a parameter a later model estimates (the `iota` case) — and keeps
  the trace a pure record of the sampled dimensions (cleaner than emitting
  redundant constant columns into a live, hot-append file). The predictive
  artifact itself carries _no_ parameter columns (it is
  `time × dims × kind × quantiles`), so the estimated/fixed distinction is a
  property of the draws, surfaced via the schema's parameter roles — which is
  also how a pairplot drops the constant columns of `draws.tsv`. There is no
  `--no-fixed` flag: the predictive output has no parameters to omit, and
  `draws.tsv` stays self-contained for re-simulation.
- **Convergence-gated, hard.** By default it _refuses_ when the fit's R̂/ESS
  indicate non-stationarity. A warning would be skimmed — the project's own
  "hard errors over warnings" rule — and the wrong figure would ship anyway. The
  refusal _is_ the feature: it names `--allow-nonconverged`, prints the max R̂
  and the offending parameters, and shows the one-liner for a provisional panel.
  `--allow-nonconverged` is the deliberate escape (the mid-run look during a 3 h
  fit), and it stamps `converged=false` plus the R̂/ESS _into the artifact_, so a
  provisional figure carries its own caveat in-band rather than in a title
  string. Every artifact — gated or not — records the convergence numbers. This
  is the single highest-leverage property; it kills audit bugs 1 and 2 at the
  source.
- **Integrates over θ.** The predictive is sampled across the posterior draws
  with the real observation model (noise included), never at a plug-in point.
- **Aligned to the observation cadence**, since camdl knows it — no
  consumer-side time-grid guessing.

### 3. The artifact (one tidy table, keyed by predictive _kind_)

Per stream, a tidy long table sharing keys with the schema, stored as a declared
CAS artifact in the run's leaf so consumers read one file instead of globbing
the store:

```
predictive/<stream>.tsv          # tidy, plot-ready, DEFAULT
  time | <dims...> | kind | q05 | q25 | q50 | q75 | q95 | log_score | crps | pit
  # kind ∈ { postpred, onestep }   extensible: kstep, lodo (#277 held-out)
  # scores NULL where undefined (postpred); populated for data-conditioned kinds

observed/<stream>.tsv            # the observed half of the panel; kind-independent
  time | <dims...> | value

predictive_samples/<stream>.parquet   # OPT-IN (--save-samples)
  time | <dims...> | kind | draw | value
```

What each file holds, in words:

- **`observed/<stream>.tsv`** — the observed half of the panel: the recorded
  value for each `(time, stratum)`, emitted by the verb in the same tidy keys as
  `predictive` (a _derived_ series from the bound data, not a copy of the source
  file), so a panel renders from the leaf without chasing the original data
  path. Independent of predictive kind — the data is the data.
- **`predictive/<stream>.tsv`** — the model's distribution over that _same_
  observable, summarized as quantiles per `(time, stratum)`: `q50` is the
  predicted median, `q05…q95` the 5–95% band a consumer draws as a ribbon. The
  `kind` column says _which_ predictive (free-forward `postpred` vs one-step
  `onestep`), so both live in one file. `log_score`/`crps`/`pit` are per-point
  calibration scores — how well the prediction matched the actual value — left
  empty for `postpred` (which is not conditioned on the data point) and filled
  for the data-conditioned kinds.
- **`predictive_samples/<stream>.parquet`** — opt-in: the raw per-draw values
  behind the quantiles, for anyone computing their own intervals.

To draw the canonical figure a consumer reads two files, joins them on
`(time, <dims>)`, and plots `observed` as points over the `predictive` ribbon,
one facet per stratum — no model knowledge, no likelihood math, no run-store
spelunking.

The shape choices:

- **`kind` is a column, not a file or a column-name prefix.** The whole purpose
  is _comparing_ predictive objects, so the set grows (one-step, k-step, the
  #277 leave-stratum-out marginal). A new predictive object is then _more rows /
  a new enum value_, never new columns or new consumer code. This is the
  unification the superseded #279 comment proposed; here it is grounded as the
  verb's output.
- **Quantile levels are columns, not long `(level, value)`.** The set is small
  and fixed and `fill_between(q05, q95)` wants columns. Tidy on the open-ended
  axes (streams, kinds), wide on the small fixed axis (quantiles).
- **Quantiles by default; per-draw samples opt-in.** Most consumers want the
  ribbon; the parquet is the escape hatch for custom intervals. This also
  retires the 35 MB-JSON-by-default of the prequential path.
- **Scores in-row, nullable** — they are per `(time, stream, kind)`; keep them
  on the row rather than re-splitting one observation across two tables.
- **Strata are rows, not files.** A stratified stream's index dims are key
  columns (`<dims...>`), so `deaths[patch, age]` is one file with `patch` and
  `age` columns. `--stream` selects at the _logical_ stream level (see
  Decisions), never a file per stratum — which also resolves the fit side's
  per-expanded-stream enumeration (F9).

### 4. `camdl fit latent <run> --stream <s>` — the third object (deferred)

Deferred this iteration: it is gated on trajectory coherence (#270 / #267)
landing, and is specced here only so the verb pair is designed as a whole.

The posterior latent state from the _conditioned_ trajectories (F20), at the
observation cadence, integrating over θ (not the median-θ plug-in of audit bug
3). This is gated on the conditioned paths being directionally valid — F21
reports S backflow and flow over-counting on a saved spatial trajectory, so the
trajectory-coherence work (#270 / #267 / #276) is a hard prerequisite: a latent
extractor over un-trustworthy paths is worse than none. Until then, `latent`
should at minimum refuse to emit flow-derived incidence it cannot reconcile with
the state deltas, rather than emit a plausible-but-wrong series.

## The workflow, end to end (UX surface for review)

The point of this section is to make the user-facing surface concrete enough to
critique. Today's reconstruction (the b2 panel script) is ~70 lines of glue;
below is the proposed workflow it collapses to.

### Today (what an analyst writes for one predicted-vs-observed panel)

```python
# ~70 lines, rebuilt in each of 12 scripts. Abridged:
tail = concat(read(t) for t in glob("…/<run>-<hash>/…/chain_*/trace.tsv"))   # path spelunking
tail = tail.filter(sweep > 250)                  # burn-in, a hand-picked literal
draws = tail.sample(150)
for k, v in FIXED.items(): draws[k] = v          # re-inject sigma/gamma/ifr/I0 by hand (footgun)
draws.write("draws.tsv")
run(["camdl","simulate",MODEL,"--draws","draws.tsv","--obs-only-dir","out/","--output-every","7"])
for p in DISTRICTS:                              # DISTRICTS hardcoded
    t = read(f"out/onset_{p}.tsv"); pivot draw×time wide; float-sort time headers
    band = percentile(t, [5,50,95])              # quantiles over (maybe non-converged) draws
# … then plot
```

### Proposed

```console
$ camdl fit run fit.toml --seed 1                          # fit — unchanged

# Predicted-vs-observed for a stream, in one command. Reads the run:
# burn-in, the fixed params, the stratum list, the obs cadence — none retyped.
$ camdl fit predict sle-8a3f12b4 --stream onset
wrote results/fits/sle-8a3f12b4/predictive/onset.tsv   (kind=postpred, 14 strata, converged=true)
wrote results/fits/sle-8a3f12b4/observed/onset.tsv

# The artifact is tidy and self-describing (schema in fit.meta.json gives index_dims):
$ head -3 results/fits/sle-8a3f12b4/predictive/onset.tsv
time  patch    kind      q05    q25    q50    q75    q95    log_score  crps  pit
7     Bo       postpred  0.0    1.0    3.0    6.0    12.0   NA         NA    NA
7     Bombali  postpred  0.0    0.0    1.0    3.0    7.0    NA         NA    NA
```

A consumer (camdl-watch, a plot script, a reviewing agent) then reads **one
file** and never touches the run store, the DSL, or the likelihood:

```python
pred = read("predictive/onset.tsv")              # facet by index_dims (from schema), ribbon from q-cols
obs  = read("observed/onset.tsv")                # overlay; join on (time, patch)
```

**The convergence gate — the error is the feature:**

```console
$ camdl fit predict sle-8a3f12b4 --stream onset
error: this fit has not converged — refusing to emit a predictive that would read as final.
  max R̂ 1.42 (rho), 1.31 (D50);  min bulk-ESS 47 (rho)   [gate: R̂ < 1.05]
  For a provisional mid-run panel, stamped into the artifact:
      camdl fit predict sle-8a3f12b4 --stream onset --allow-nonconverged

$ camdl fit predict sle-8a3f12b4 --stream onset --allow-nonconverged
warning: fit has not converged (max R̂ 1.42); artifact stamped converged=false.
wrote results/fits/sle-8a3f12b4/predictive/onset.tsv   (converged=false, max_rhat=1.42)
```

**Comparing predictive objects — one table, `kind` rows:**

```console
$ camdl fit predict sle-8a3f12b4 --stream onset --kind onestep   # adds onestep rows; scores populated
# predictive/onset.tsv now carries postpred + onestep; a consumer does filter(kind==…) to compare.
```

**Re-simulating a raw trace standalone (#273, fill-missing-only):**

```console
$ camdl simulate model.camdl --draws tail.tsv --fit fit.toml --obs-only-dir out/
# fills sigma/gamma/ifr/I0 from [fixed]; never overwrites a column present in tail.tsv.
```

**Pairplots omit fixed params by role, not by guessing:**

```python
roles = read_json("fit.meta.json")["schema"]["parameters"]
pairplot(draws, cols=roles["estimated"])         # roles["fixed"] = {sigma: 0.11, …} dropped by role
```

## Execution sequence

Ordered by dependency; early items are independently shippable.

1. **#273 — fill missing params from the fit (ship now).**
   `simulate --draws <tsv> --fit <run>` backfills any parameter _absent_ from
   the draws source from `[fixed]`, never overwriting a present column (#273's
   lead recommendation, over its option #2 of fattening `trace.tsv` with
   constant columns). Closes the trace/draws re-simulation asymmetry and the
   fixed-map footgun; the trace stays a pure record of sampled dimensions.
   Design-stable and independent of the rest — the prerequisite the verb's
   draw-resolution builds on.
2. **Posterior-from-run resolution.** `--draws posterior --fit <run>` resolves
   the post-warmup tail using the fit's `burn_in`. Removes the trace-glob /
   burn-in-literal steps. Composes with (1).
3. **The schema descriptor (#279 Part 1).** Emit streams × dims × levels ×
   value_kind × likelihood per fit. Independent; also fixes the consumer's
   stratum-list duplication and the watcher's family-grouping heuristic.
4. **`camdl fit predict` + the `predictive/<stream>.tsv` artifact, free-forward
   `kind=postpred` first.** The verb assembles (1)+(2), is convergence-gated,
   and writes the tidy CAS artifact. Collapses the twelve scripts. The headline.
5. **Unify `kind=onestep`.** Emit the gh#269 per-stream prequential into the
   same table (it already lands in nearly this shape:
   `t, stream, y_obs,
   y_pred_q05…q95, log_score, crps, pit, ess`). Now one
   file carries both objects.
6. **`camdl fit latent` (F20)** — gated on trajectory coherence
   (#270/#267/#276).
7. **Metrics + held-out (#277 / F18).** WAIC / PSIS-LOO / PIT-coverage scored on
   the unified table; `kind=lodo` rows from leave-stratum-out. This is the
   model-criticism proposal's surface, consuming this proposal's artifact.

Steps 1–3 are shippable in parallel today. Step 4 is the design-bearing one and
the point at which the consumer's twelve scripts collapse to one command.

## Relationship to existing issues and proposals

- **#273** is execution step 1 — the foundation, not a serial blocker; its
  design is independent of the verb.
- **#279** splits cleanly: its Part 1 (schema descriptor) is step 3 here; its
  Part 2 (the postpred artifact) is the `kind=postpred` slice of step 4; its
  Part 3 (geometry) is an orthogonal forward-looking extension that this
  artifact's dimension keys are designed to carry later.
- **#277** is downstream: the model-criticism metrics and held-out evaluation
  score _this_ artifact. The `2026-06-20-model-criticism-outputs.md` proposal's
  open questions on routing `simulate --obs` through a single long writer are
  answered here — the predict verb's tidy table is that writer's output.
- **gh#269** (one-step prequential) is the `kind=onestep` source; **gh#86**
  (`--draws prior`) is already landed and retires the hand-rolled prior sampler.

## Decisions

- **Verb home.** `camdl fit predict` and `camdl fit latent` are fit subcommands
  — run-resolution and the convergence gate are the point, and they belong to a
  verb that knows it operates on a _run_. The `--draws posterior --fit <run>`
  resolver (step 2) is the shared layer underneath; `simulate --draws … --obs`
  stays as the params-level entry point on the **same** engine and writer. The
  overlap is two documented entry points over one predictive path — not a forked
  obs-output path (the §A2 hazard the model-criticism proposal flags).
- **Convergence gate: hard error by default**, `--allow-nonconverged` escapes
  and stamps `converged=false` + R̂/ESS into the artifact (§2). The gate **reuses
  the fit's existing R̂ test** — `max(Â) < gate.a_thresh` (default 1.01,
  per-fit-configurable; `config_v2.rs`) — not a new threshold or a second
  "converged" verdict.
- **Schema and parameter roles live in `fit.meta.json`** (§1) — one file,
  already read by consumers; no separate `schema.json`.
- **Run reference.** `--fit fit.toml` is primary; a run id (`sle-8a3f12b4`) is
  also accepted, matching `fit summary` / `show`.
- **Artifact location.** The artifacts sit in the **stage dir alongside
  `draws.tsv`** (`<stage>/predictive/<stream>.tsv`,
  `<stage>/observed/<stream>.tsv`, opt-in `predictive_samples/`) — the same
  scheme the posterior draws already use.
- **Quantile set.** Fixed `{05, 25, 50, 75, 95}` by default (schema-stable
  across runs), with an opt-in override.
- **`observed/<stream>.tsv` is a _derived_ series, not a mirror.** It is the
  panel's observed half, emitted by the verb in the same `(time, <dims>, value)`
  keys as `predictive` — needed because the panel needs observed points and the
  verb has them from the bound data. (For contrast: `fit.toml` is archived as
  `fit.toml.original`; the model is _not_ copied into the leaf, being
  hash-pinned and recompilable. The observed series is emitted, not copied.)
- **Stream selection.** `--stream` names the **logical** stream and accepts
  several; bare `fit predict <run>` emits **all** logical streams. A stratified
  stream is **one** file with its index dims as key columns — `deaths[patch]` →
  `time | patch | kind | q…`, `deaths[patch, age]` →
  `time | patch | age | kind | q…` — never a file per stratum. This relies on
  §1's schema to map a logical stream to its expanded members (the IR is fully
  expanded), and resolves the fit-side asymmetry where `[data.observations]`
  must enumerate `cases_Bunia, …` by hand (F9).
- **Latent (`camdl fit latent`) is deferred** until trajectory coherence (#270 /
  #267) lands (§4).

## Still open

The design is settled; what remains is implementation detail — the
override-quantile syntax and the per-draw samples file format. Deferred by
decision: geometry (#279 Part 3, orthogonal), `latent` (§4, gated on #270 /
#267), and the scoring metrics (#277, a downstream consumer of this artifact).
