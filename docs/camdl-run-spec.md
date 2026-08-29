# camdl Run System Specification

**Version:** 0.5-draft\
**Date:** 2026-08-11

> **Scope:** This spec describes camdl's run system: forward simulation (single
> runs, multi-cell ensembles, and `batch run` sweeps), inference pipelines
> (`fit.toml`), parameter sweeps, predictive workflows, and the
> content-addressed store that holds every result together with its provenance.

---

## 1. Design Principles

### 1.1 Background: How camdl Partitions Model Inputs

A camdl model defines a stochastic simulator whose inputs are partitioned into
three categories:

- **Model parameters (M):** the tuneable knobs — quantities that _could_ be
  varied during calibration or sensitivity analysis. Transmission rate β,
  recovery rate γ, reporting probability ρ. Declared in the `.camdl` file's
  `parameters { }` block.

- **Configuration (C):** structural and runtime choices that are never subject
  to calibration. Population structure, integrator and step size, the output
  schedule, which interventions are enabled. Defined by the `.camdl` model plus
  the CLI/config knobs that select a backend and a `dt`.

- **Seed (s ∈ S):** the base random seed for stochastic simulation. Always a CLI
  or config-file argument, never baked into the model.

A simulation is the mapping Sim(m, c, s) → y, producing a trajectory and
(optionally) sampled observations. Every workflow here — forward simulation,
sweeps, inference, predictive checks — is an operation on these three layers.

**Scenarios (σ)** are deterministic patches that modify parameters and/or
configuration from their baseline values: σ(m, c) → (m′, c′). They are declared
in the `.camdl` file's `scenarios { }` block (compiled to `model.presets` in the
IR) and selected at runtime with `--scenario`, or constructed ad hoc on the CLI
with `--enable` / `--disable`. A scenario that patches nothing is the baseline.

**Inference** operates on a _view_ of the parameter space. When fitting, some
parameters are estimated and others are held fixed. That partition —
`[estimate]` versus `[fixed]` in `fit.toml` — defines which parameters the
algorithm explores and which it treats as known constants.

### 1.2 File Roles and Separation of Concerns

**One file per concern, no overlap.**

```
model.camdl      → what the model IS (structure, priors, scenarios)
params.toml      → a point m ∈ M (concrete parameter values)
fit.toml         → how inference RUNS (what to estimate, priors, stages, data)
batch TOML       → how a batch RUNS (sweep/design/scenarios/seeds)
```

The `.camdl` file owns structure. It also owns two things that are easy to
mislocate: parameter **priors** (`p ~ Normal(0, 1)` in the `parameters` block)
and named **scenario presets**. It does not own concrete parameter point values
outside a preset.

| Concern               | Declared in                                 | Overridable by                                     |
| --------------------- | ------------------------------------------- | -------------------------------------------------- |
| Model structure       | `.camdl` only                               | never                                              |
| Parameter names/types | `.camdl` only                               | never                                              |
| Parameter values (M)  | `.camdl` default, params TOML, `[fixed]`    | draw/sweep point, scenario, `--param`/`--fixed`    |
| Scenarios (σ)         | `.camdl scenarios { }`                      | CLI `--enable`/`--disable`, batch `[[scenario]]`   |
| Interventions         | `.camdl` only                               | scenarios and `--enable`/`--disable` toggle them   |
| Priors                | `.camdl` `~`, `fit.toml [estimate.p.prior]` | fit.toml wins over the model                       |
| Backend / `dt`        | CLI, batch `[config]`, fit.toml stage       | `--backend` / `--dt` on the simulate-side commands |
| Seeds                 | CLI, batch `[config.seeds]`, `fit_seeds`    | `--seed` / `--seeds`                               |
| Sweep / design        | CLI `--sweep`, batch `[sweep]`/`[design.*]` | never                                              |
| Estimate vs fixed     | `fit.toml` only                             | `fit run --sweep` varies a `[fixed]` parameter     |
| Which stages run      | `fit.toml [stages.*]`                       | `fit run --stage NAME` selects one                 |

**The model file is self-contained for a single run** provided every parameter
gets a value from somewhere: a `.camdl` default, a params TOML, a scenario, or a
`--param`. A parameter with no value from any source is a hard error naming the
parameter and the sources that were consulted.

### 1.3 Precedence Chains

Parameter resolution for the simulate-side commands (`simulate`, `batch run`,
`pfilter`, `survey`, `profile`, `lineage`) is a single function,
`params_resolver::resolve_parameters`, which is the only writer of
`model.parameters[i].value` outside the IR layer. It layers six tiers, last
wins:

```
1.  model default            (`p.value` from the .camdl declaration)
2.  fit.toml [fixed]         (inference commands, and --fit on others)
3.  --params / --fixed-file  (each file layered in order)
3.5 draw row or sweep point  (automated M-layer variation)
4.  scenario                 (preset `params` + multiplicative `scale`,
                              or an inline ad-hoc `set`/`scale`)
5.  --param / --fixed        (the user's explicit assertion; highest)
```

The structural distinction between 3.5 and 5: a draw or sweep value is
_automated_ M-layer variation and is counterfactual-modifiable, so a scenario
patch (tier 4) overrides it. A `--param`/`--fixed` value is the user's explicit
assertion about this specific run and overrides everything, scenario included. A
named preset and an equivalent inline ad-hoc scenario resolve at the same tier,
so they are indistinguishable in the resolved value — only the recorded
provenance label differs.

`[estimate]` membership is narrowed by the same walk: a parameter named by a
user-explicit `--fixed`/`--fixed-file` is removed from the estimated set (with a
warning), while a scenario patch or a draw/sweep value never kicks a parameter
out of `[estimate]`.

**Priors** resolve on their own three-tier chain (`fit.toml [estimate.p.prior]`
→ the model IR's `~` declaration → flat). For `camdl profile` the flat tier is a
warned fallback; for `camdl fit run` it is reachable only by writing
`prior = { flat = {} }` explicitly — an implicit fall-through to a flat prior is
a hard validation error before the fit starts, and the resolved source per
parameter is recorded in `fit.meta.json`.

### 1.4 Core Design Rules

**One job type behind both front-ends.** `camdl simulate` and `camdl batch run`
build the same `SimulateJob` (`sim_job.rs`) and hand it to one engine
(`engine::run_job`), which plans the `param-point × scenario × seed-slot` grid
and drives it through a `RunSink`. The two commands differ only in their sink:
`simulate` writes per-cell store leaves plus a combined wide-format mirror;
`batch run` writes per-cell store leaves only. `SimulateJob` deliberately has no
serde derive — the front-ends each parse their own surface (clap args, batch
TOML) and converge on the resolved struct, rather than sharing a second wire
schema that would drift from the batch TOML's own versioned form.

**Every parameter choice is accounted for, but a model default is a legal
source.** In `fit.toml` the rule is exhaustive and enforced: every model
parameter must appear in exactly one of `[estimate]` or `[fixed]`; a parameter
in both, in neither, or in neither-and-not-in-the-model is a hard error before
the fit starts. On the simulate side the rule is weaker: the resolver walks the
six tiers above and errors only if _no_ tier supplied a value. A `.camdl`
parameter declared with a default therefore runs without a params file, and the
recorded provenance says `model_default`.

**Sweeps are orthogonal to everything.** A sweep is "run this thing at multiple
parameter values," and it works on both sides. `batch run` takes `[sweep]` (a
deterministic grid) or `[design.*]` (space-filling), mutually exclusive.
`fit run --sweep NAME=SPEC` varies a `[fixed]` parameter, where SPEC is
`V1,V2,…` | `lin(min,max,n)` | `log10(min,max,n)`; sweeping a parameter that is
in `[estimate]`, or not in `[fixed]`, is an error. On the simulate side, sweeps
compose with scenarios and seeds by Cartesian product.

**Draws and sweeps are different operations.** A sweep is a deterministic grid
the user designed; draws are samples from a distribution (posterior, prior, or
uniform over bounds). They have different provenance and different downstream
semantics, so they are separate variants of a sum type — `ParamSource::Point`,
`::Sweep`, `::Draws` — never conflated. `ParamSource::Draws` further records
whether the rows came from a user-authored file, because a scenario that patches
a parameter the file also supplies is a hard error for an explicit file (two
pinnings of θ, ambiguous intent) and a silent scenario win for generated draws.

**Provenance is structural.** Every stored run carries a `run.json` recording
its identity, its inputs, its lineage, its file manifest, and the column schema
of its tabular outputs. Nothing is overwritten in place: a changed input
produces a different identity and therefore a different directory, so a stale
result is never silently served under a new configuration.

**Reproducibility is structural.** Every stored artifact is content-addressed by
the resolved inputs that produced it. Same inputs → same `run_id`. M and σ stay
distinct at the CLI (`--param` operates on M;
`--scenario`/`--enable`/`--disable` operate on σ) and in the identity (they are
separate hash levels), so a counterfactual can never be confused with a
parameter edit.

---

## 2. Project Directory Structure

A camdl project is a convention, not a scaffold camdl creates or checks. The
layout the vignettes and book chapters use is:

```
project/
├── models/          # .camdl sources
├── data/            # observation TSVs, lookup tables
├── params/          # parameter-point TOMLs
├── fits/            # fit.toml configs
├── batches/         # batch TOML manifests
└── results/         # the content-addressed store (see below)
```

Only `results/` has meaning to camdl. Its location resolves with the precedence
**CLI `--output-dir` > config-file `output_dir` > `CAMDL_OUTPUT_DIR` >
`./results`** (`run_paths::output_root`). Project-specific config sits above
ambient shell state deliberately: setting `CAMDL_OUTPUT_DIR` to keep scratch
runs separate must not silently redirect a fit.toml that declared
`output_dir = "results/he2010"`.

### 2.1 The content-addressed store

Every result camdl keeps lives in one store under the output root, partitioned
by artifact kind:

```
results/
├── index.json            # rebuildable lookup cache (§2.7)
├── .staging/             # atomic-commit scratch (§2.7)
├── sims/                 # one leaf per simulated cell
├── ensembles/            # combined wide-format TSV across a multi-cell simulate
├── fits/                 # fit segments, each holding its stage leaves
├── pfilters/             # standalone particle-filter loglik evaluations
├── surveys/              # likelihood-landscape scans
└── profiles/             # profile-likelihood grid points
```

`obs/` and `projections/` are reserved partitions: their store directory and
run-id tag are pinned but no leaf is emitted for them yet.

**Identity is factored.** A leaf's identity is the ordered tuple of per-level
content hashes along its path, and the store path is a readable nested
_factoring_ of that tuple rather than a flat blob directory. Each path segment
is `{label}-{hash8}`: the **label is provenance** — renaming it yields a new
directory, i.e. a harmless cache miss, never a wrong answer — and the **`hash8`
is identity**, the first 4 bytes of that level's SHA-256 rendered as 8 hex
characters. The leaf's address is

```
run_id = SHA256( HASH_VERSION ++ kind_tag ++ level_count ++ [level hashes in path order] )
```

with the kind folded in as a fixed-width enum index and the level list
length-prefixed, so two kinds with a coincidentally equal level sequence cannot
alias and `([h1,h2], [h1])` cannot collide with a concatenation. The full 64-hex
`run_id` and the full per-level hashes are recorded in `run.json`.

The levels per kind:

| kind            | partition   | levels, in path order                     |
| --------------- | ----------- | ----------------------------------------- |
| `sim`           | `sims`      | model · config · params · scenario · seed |
| `sim_ensemble`  | `ensembles` | model · config · params · grid            |
| `fit_stage`     | `fits`      | fit · stage · seed                        |
| `pfilter`       | `pfilters`  | model · config · params · seed            |
| `survey`        | `surveys`   | model · config · box · seed               |
| `profile_point` | `profiles`  | profile · point · stage · seed · start    |

**The one rule for consumers: resolve runs by reading `run.json`, never by
parsing path segments.** Path segments mirror the `levels` array for human
navigation only. Do not infer kind, parameters, seed, or lineage from a path.

Two mechanics constrain the segment strings. A label is lowercased and every
character outside `[a-z0-9._-]` is mapped to `_`, so hyphens and dots survive
(`chain_binomial-dt1`, `01-scout`, `seed_42`); a label longer than 200 bytes is
truncated to a prefix and suffixed `..{hash16}` so it fits inside the POSIX
255-byte `NAME_MAX`. If two distinct leaves would land on the same directory
name because their short hashes collide at every level, the later one's final
segment gets a `~{disambiguator}` suffix, escalating `~{hash16}` → `~{full64}`
until unique. A reader enumerates sibling directories and reads each `run.json`
rather than reconstructing an expected name.

**Presentation is stripped before hashing.** IR fields that only change how a
result is _rendered_ — `output.format`, `simulation.time_semantics` — are
normalized out inside `ModelDigest::from_model`, which sits on every identity
path, so a new artifact kind cannot silently opt out of the strip. A field that
changes _which values are computed or stored_ is identity and must re-key; a
re-encoding of the same values is presentation and must not. Re-keys are
deliberate, version-bumped events: a per-struct `schema_version`, the crate-wide
`HASH_VERSION`, or `ir/VERSION`.

### 2.2 Fit Result Layout

A fit occupies a **segment** directory `results/fits/{stem}-{fit_hash8}/`, where
`stem` is the slugged basename of the `fit.toml` and `fit_hash8` is the
fit-level content hash. The segment has **no `run.json` of its own** — it is a
path level, not a leaf. Under it, one leaf per (stage × fit seed):

```
results/fits/{stem}-{fit_hash8}/
  fit.meta.json                     # fit-wide provenance sidecar (§2.2.1)
  fit.toml.original                 # archived producing config
  model.camdl.original              # archived model source (when given as .camdl)
  model.ir.json                     # archived compiled IR
  model.render.json                 # archived display render (when given as .camdl)
  model.graph.json                  # archived flow graph (when given as .camdl)
  sweep_failures.tsv                # gate failures across a --sweep grid, when any
  synthetic/truth.toml              # [synthetic] fits only
  synthetic/data/ds_NN.tsv          # [synthetic] fits only, one per sim seed
  predictive/<stream>.tsv           # `fit predict` output (§2.2.2)
  predictive.json
  observed/<stream>.tsv
  observed.json
  quantities/<name>.tsv             # `quantities {}` sidecar, when requested
  quantities.json
  {NN}-{stage}-{stage_hash8}/       # one per stage, NN = topological ordinal
    seed_{n}-{seed_hash8}/          # one per fit seed — the leaf
      run.json
      ...method-specific artifacts...
```

The archived files and the `predictive`/`observed`/`quantities` outputs are
**regenerated sidecars**: they are not part of any leaf's file manifest, carry
no `run_id`, and are overwritten in place.

**What the fit hash covers.** The fit level is a `FitDigest` over the whole
model IR digest (which itself folds `ir/VERSION` and the engine version), the
content digests of every training data stream, the content digests of every
explicit `[data.holdout]` stream, the engine version, and the canonical JSON of
the resolved fit config with three slices normalized out: `stages` (each stage
owns its own block at the stage level), `fit_seeds` (the seed level owns the
seed), and `output_dir` (pure write-location provenance). Consequently **editing
a `[stages.*]` block does not move the fit segment** — it re-keys only that
stage's leaf, which is what lets you retune a posterior stage without
invalidating the scout that feeds it. Two `fit.toml` files that differ only in
stage blocks share one fit hash and are separated on disk only by their
differing stems.

Data files enter by **content**, not path: rewriting a holdout or training TSV
in place re-keys the fit, so a stale held-out score cannot be reused. The same
holds for chain-start sources — `--posterior`'s `draws.tsv`, `--params`' TOML,
and a `survey_top_k` landscape all fold their file digest into the stage's
`deps`.

**Stage levels and lineage.** The stage level hashes the stage's identity
payload (algorithm and its settings), the number of stored posterior
trajectories, the target chain length, the _resolved_ observation alignment
(`exact` or `snap`, per algorithm), and the stage's `deps` — the upstream
artifacts it consumes. Folding `deps` in is what makes cross-stage invalidation
work: `02-posterior`'s hash contains `01-scout`'s identity, so regenerating the
scout re-keys the posterior. The stage's path label carries a zero-padded
topological ordinal (`01-scout`, `02-posterior`) so execution order sorts
lexicographically.

**Leaf contents by method.** An IF2 stage writes `fit_state.toml`,
`mle_params.toml`, `final_params.toml`, `chain_starts.tsv`,
`chain_evaluations.tsv`, `diagnostics.tsv`, and per chain
`chain_N/parameter_traces.tsv` + `chain_N/final_params.toml`. A PGAS or PMMH
stage writes `fit_state.toml`, `draws.tsv` (the thinned post-warm-up cloud),
`diagnostics.json`, and per chain `chain_N/trace.tsv`,
`chain_N/resume_state.bin`, and — when posterior trajectories are requested —
`chain_N/trajectories.tsv` with a `chain_N/trajectories.json` manifest.

`mle_params.toml` is written params-first with a trailing `[provenance]` table,
so a bare `toml` reader still sees the parameters at the top level. The block
records `camdl_version`, `timestamp`, `content_hash` (a tamper hash over the
numeric values only, at 12-decimal precision, so editing a provenance field does
not invalidate it), `fit_hash`, `backend`, `dt`, `model`, `model_identity`,
per-stream `[provenance.data]` path+hash entries, `seed`, `stage`, 1-indexed
`chain`, `log_likelihood`, `loglik_sd`, `n_particles`, and optional
`ess_mean`/`ess_min`. `camdl simulate --params` reads `backend`/`dt` back to
warn when a downstream simulation would run under different dynamics than the
fit.

#### 2.2.1 `fit.meta.json` — fit-level sidecar

The fit segment carries a `fit.meta.json` holding the fit-wide provenance that
has no `RunRecord` of its own. It is **derived provenance, never a source of
truth**: everything in it is a readable projection of inputs already hashed into
the leaf identities, written after identity is fixed and never fed back into any
hash. Its keys serialize in sorted order (the maps are `BTreeMap`s) so two
identical runs produce byte-identical sidecars and a diff of two runs' metadata
is meaningful.

Fields: `label` (the sticky user `--label`; a later stage-only re-run that
passes no label preserves the one already on disk), `model_path`,
`model_identity` (the hex structural model hash), `fit_toml_path` (where the
producing config lived — the directory its relative `[model]` / `[data]` paths
resolve against when the config is recovered from the segment), `fit_toml_hash`
(SHA-256 of the producing config's raw bytes — provenance, i.e. which exact
bytes produced this fit; a `fit.toml` handle is NOT looked up by it, see below),
`data_hashes`, `estimated`, `fixed`, `resolved_priors` (one `{param, source}`
entry per estimated parameter, `source ∈ {fit_toml,
model_ir, flat_explicit}`,
emitted only when the fit has a Bayesian stage), `schema` (the
observation/dimension schema below), and `docs` (the model's `#'` documentation
dictionary, keyed by declaration name, so a consumer can label any output
column).

Two caveats a consumer must know. `data_hashes` is built from the explicit
`[data.observations]` map only, so a fit that uses the `[data] file = "..."`
shorthand records an empty map here even though the identity path digests the
expanded per-stream set — read the leaf's identity, not this field, to answer
"which data". And `fixed` is projected from the base configuration, so under
`fit run --sweep` every swept segment records the _unswept_ value of the swept
parameter; the segment's fit-level hash, not this field, distinguishes the sweep
points. `parameters_provenance` is reserved and always empty.

A third thing to know, because it is the one field whose name invites the wrong
inference. `fit_toml_hash` answers _provenance_ — which exact bytes produced
this fit — and nothing resolves a handle by it. Handing a config to a verb
(`camdl compare fit.toml`, `camdl fit summary fit.toml`) asks a different
question: does this config MEAN what the stored fit was run from? That is
answered by canonicalising the parsed value tree — comments, whitespace, table
order, and the spelling of a float all discarded, `[stages.*]` / `[estimate]` /
`[data.observations]` order preserved because camdl reads those in order — and
comparing it against the same canonicalisation of the segment's
`fit.toml.original`. The archive is read at lookup time, so a fit stored before
this rule existed resolves under it unchanged. Reflowing a comment changes
`fit_toml_hash` and leaves the lookup identity alone; changing a particle count
or a data path changes the identity and the config no longer resolves to that
run.

The `schema` block lets a consumer facet any stream and label panels with no DSL
parsing:

```json
{
  "schema": {
    "dimensions": { "patch": { "levels": ["Bo", "Bombali"] } },
    "streams": [
      {
        "name": "onset",
        "index_dims": ["patch"],
        "value_column": "onset",
        "value_kind": "count",
        "likelihood": "neg_binomial"
      }
    ]
  }
}
```

`dimensions` maps each indexing dimension to its ordered levels (the union over
all streams, in first-appearance order). `streams` carries one descriptor per
**logical** stream, grouped by the `from <label>` data-source key, so a
stratified `onset[patch]` is one entry with `index_dims = ["patch"]` and never
one entry per expanded leaf. Each descriptor gives `name`, `index_dims`,
`value_column` (the scored `~` left-hand side), `value_kind` (the DSL role —
`count` / `real` / `probability`; omitted when the model declares no
`columns {}` block), and `likelihood` (the family tag). The schema is a pure
fold over the same observation leaves the particle filter binds, so it cannot
disagree with what was fit.

#### 2.2.2 `fit predict` predictive artifact

`camdl fit predict <fit>` resolves the fit handle — `@label`, a `fit.toml` path,
a run directory, or a `fit_hash` prefix — to its segment and writes two tidy,
plot-ready families of files there, plus a JSON manifest for each.

```
results/fits/<stem>-<hash8>/predictive/<stream>.tsv
  scenario | sweep:<param>… | time | <dims…> | horizon | treatment
           | rhat_max | ess_min | n_draws | q05 | q25 | q50 | q75 | q95

results/fits/<stem>-<hash8>/observed/<stream>.tsv
  time | <dims…> | value
```

`predictive/<stream>.tsv` summarizes the model's distribution over the
observable as quantiles of sampled `y_rep` per
`(scenario, sweep point, time, stratum)`. Every axis is a column, so a new
predictive cell is more rows and never new consumer code. The `horizon` column
is `free_forward` (run the fitted model forward from the start, `p(y_t | θ)`) or
`one_step` (`p(y_t | y_{1:t-1})`, re-running a bootstrap filter per posterior
draw and pooling over particles × draws); both stack under one header. The
`treatment` column is `posterior` when the band averages over the whole draw
cloud — the only treatment the band-builder accepts today, enforced by typing
rather than a runtime check, so a posterior-labelled band over a single point
estimate is unrepresentable — with `plug_in` reserved. `rhat_max`/`ess_min`
carry the producing stage's convergence numbers, empty when the stage reported
none; `n_draws` is the cloud size behind each band. The `scenario` column is
`fitted` for the one-step rows, which are scenario- and sweep-agnostic by
construction: filtering _observed_ data through a counterfactual model is
ill-defined.

`observed/<stream>.tsv` is the recorded value per `(time, stratum)` in the same
tidy keys. A hole — a scheduled but unobserved cell — renders as an empty
`value`, distinct from an observed zero. A consumer reads both files, joins on
`(time, <dims>)`, and plots `observed` over the `predictive` ribbon, one facet
per stratum, using `index_dims` from the `fit.meta.json` schema and no
run-store, DSL, or likelihood knowledge.

`predictive.json` (schema tag `camdl.predictive/v1`) declares, per stream, which
columns are join coordinates versus band versus per-cell diagnostics, the
`value_kind`, and the band's quantile levels; it also carries a
`chain_selection` block when `--exclude-chains` narrowed the cloud, so a
chain-subset band is never mistakable for a full-cloud one. `observed.json`
(`camdl.observed/v1`) is its sibling for the observed series.

**Calendar semantics travel with the artifact.** `predictive.json`,
`observed.json`, `quantities.json`, and the per-chain `trajectories.json` each
carry a top-level `calendar` block, produced by one shared emitter so every
exporter agrees:

```jsonc
"calendar": { "time_unit": "days", "origin": "1910-01-01", "days_per_unit": 1.0 }
```

`time_unit` is the model's (`days` / `weeks` / `months` / `years`); `origin` is
the model's ISO `YYYY-MM-DD` origin, or `null` for a bare-numeric-time model —
the signal to plot `time` numerically. `days_per_unit` is the exact length of
one `time_unit` in days (`days`=1, `weeks`=7, `months`=365.2425/12,
`years`=365.2425), exported so a consumer converts with no hardcoded unit table.
A consumer maps `time → date` as `origin + time · days_per_unit` **days**. A
calendar-anchored model is constrained to `days`/`weeks` (camdl rejects `origin`
with `months`/`years`, E320), so wherever there is an `origin` to map,
`days_per_unit` is 1 or 7 and the mapping is exact and identical to camdl's own
rendered `date` column. The model's internal `origin_rata_die` is deliberately
not exported: it is keyed to camdl's own epoch and would mislead a consumer that
read it as a Julian or Rata day number. Numeric `time` stays the canonical,
diff-stable axis; `calendar` is additive metadata, not a re-encoding of it.

#### 2.2.3 Which object a quantity is read on

`fit predict` also writes the model's `quantities {}` block to
`quantities/<name>.tsv` plus a `quantities.json` manifest. Those numbers are not
all folded over the same object, and the manifest says which.

A quantity anchored **inside the observed record** — `value_at(EXPR, last_obs)`,
`value_at(EXPR, first_obs + 2 'weeks)`, or a literal time at or before the last
observation — is a retrospective estimand: there are observations covering the
instant it reads, so it is folded over that draw's **conditioned smoothing
path** `p(x | y, θ)`, saved by the fit at `chain_N/trajectories.tsv`. Everything
else is folded over the **free-forward replay** the predictive band is built
from: a reduction with no anchor (`final`, `max`, `mean`, `time_of_max`,
`integral`, a threshold crossing) is a property of a whole simulated path, and
an anchor past the last observation (`last_obs + 8 'weeks`) is a projection.

Each manifest entry carries the tag:

```jsonc
{ "name": "outbreak_size", "evaluated_on": "smoothed", "n_conditioned_draws": 300 }
{ "name": "peak_infectious", "evaluated_on": "replay" }
```

`evaluated_on` is `smoothed`, `replay`, or `replay_unconditioned` — the last for
a quantity that reduces `observations.<stream>` at an in-window anchor, which
stays on the replay because a saved path carries the conditioned projection
(`inc_<stream>`, a mean), not a draw from it. That case is announced on stderr.

Three consequences a reader has to know:

- **Only the saved subset of draws has a conditioned path.** The trajectory save
  stride (`n_trajectories`) and `thin` need not agree, so a smoothed quantity
  bands over the draws that have one; the rest are **censored**, counted in
  `n_censored`, never given the replay's answer. `n_conditioned_draws` is the
  band's real denominator, and the count is also printed on stderr.
- **A counterfactual arm keeps the replay.** A `--scenario` / `--sweep` cell, or
  a run carrying `--enable`/`--disable`, replays a different model than the one
  the smoothing path was inferred under, and the data a counterfactual would
  have generated do not exist. Those cells are tagged `replay`.
- **A fit that saved no paths reports the quantity as fully censored**, with a
  named note on stderr, rather than falling back to the replay.

### 2.3 Sweeping a fit over a fixed parameter

`camdl fit run <config> --sweep NAME=SPEC` applies each sweep point to the
resolved `[fixed]` block before identity is computed. A sweep point is therefore
a different fit — different `[fixed]` values give a different `FitDigest` — and
each point gets **its own fit segment**, a sibling directory sharing the stem:

```
results/fits/03_rho_sweep-{h8_for_rho_0.5}/01-scout-{h8}/seed_1-{h8}/
results/fits/03_rho_sweep-{h8_for_rho_0.7}/01-scout-{h8}/seed_1-{h8}/
```

There is no nesting of sweep points under a shared fit directory: within a
segment, the layout of a swept fit is identical to that of an unswept one. When
one or more sweep cells fail their convergence gate, a `sweep_failures.tsv`
(`cell`, `sweep_point`, `sweep_values`, `stage`, `reason`) is written to the
segment for the _unswept_ base configuration.

### 2.3.1 Concurrency: the leaf lock

The store enforces concurrency at the leaf, in two write modes.

**Mode A (atomic)** — used by `sim` and `sim_ensemble`, whose whole output is
known in memory before the commit. The leaf is written into a per-attempt
staging directory under `<root>/.staging/`, fsynced in a fixed order (each
artifact, then `run.json`, then the directory), then moved into place by
`rename`. The staging name carries the process id and a process-local counter,
so two concurrent same-identity commits never share a staging directory.

**Mode B (streaming)** — used by fit stages, `pfilter`, `survey`, and profile
points, which write into the leaf as they go and whose display summary (a
loglik, a landscape score) is only known at the end. The writer creates `.lock`
with `O_EXCL` inside the leaf directory, writes a `Running` `run.json`, streams
its files, and commits by flipping status to `Completed` and building the file
manifest from what it actually wrote. A concurrent claimant that finds a live
`.lock` gets `CasError::FitInProgress` naming the holding pid.

A crashed writer leaves its `.lock` and a `Running` `run.json` behind. A re-run
reclaims such a leaf: it liveness-checks the recorded pid (`kill(pid, 0)` on
unix), and a dead holder's lock is removed and the leaf re-claimed. Reclaim is
serialized through a second `.reclaim` lock — only the `.reclaim` holder may
remove `.lock`, and it re-confirms the holder is dead while holding `.reclaim`,
so two processes can never both delete the lock and both proceed. On a non-unix
platform the pid cannot be checked, so a held lock always reports
`FitInProgress`. As defense in depth, a claimant that wins the lock and then
finds a `Completed` record refuses to clear it (`ReclaimRaceCompleted`) rather
than blind-wiping a finished result.

**Cache semantics.** `lookup` reports one of four outcomes for a path against an
expected identity: `Hit` (identity matches, status `Completed`, file manifest
intact), `Miss` (no `run.json`), `Stale` with a reason (`Incomplete` — not
completed or unparseable; `Corrupt` — a listed file is missing or its size/mtime
changed; `OrphanFiles` — an undeclared file or directory is present;
`SchemaDrift` — a stale `hash_version`/`format_version`), or `Collision` (a
_different_ full identity occupies this path via a short-hash collision, in
which case the incumbent is never touched and the newcomer is disambiguated).
The integrity gate is deliberately cheap — presence, size, and mtime, no
re-digest — so listing a fit with many large chain files does not re-hash them;
the recorded per-file SHA-256 is there for integrity tooling, and no read path
recomputes it today. The manifest is an exact set over the leaf's whole subtree,
except reserved files and _declared children_: a declared child subdirectory
(`obs/`) is a boundary, recognized and not recursed into.

**Which commands skip on a cache hit.** `batch run`, `fit run`, and `profile`
consult `lookup` and skip the computation on a `Hit` (unless `--force`).
`camdl simulate` does not: it runs every planned cell and relies on the commit
being idempotent, because its combined wide-format mirror needs every cell's
rows.

### 2.4 Simulation Result Layout

A single-cell `simulate` writes one `sim` leaf:

```
results/sims/{model}-{h8}/{backend}-dt{dt}-{h8}/{params}-{h8}/{scenario}-{h8}/seed_{n}-{h8}/
  run.json
  traj.tsv
  event_log.tsv          # with --event-log on a LINEAGES-capable backend
  reactive_log.tsv       # when a reactive policy was active
  obs/{obs_h8}-{obs_seed}/{stream}.tsv + obs.json    # declared child, with --obs
```

The five levels hash disjoint slices of the resolved input set. **model** is the
presentation-normalized whole-IR digest plus `ir/VERSION` and the engine
version. **config** is backend, `dt`, `t_start`, `t_end`, the output schedule,
`allow_degenerate_rates`, and the trajectory column view (`--no-flows`,
`--columns`); `--output-every` is not here because it is lowered into the output
schedule upstream. **params** is the resolved base name→value map plus the
digests of any `--table` files. **scenario** is the enabled and disabled
intervention sets plus the parameter patch. **seed** is the resolved
`process_seed` and the base `--seed`.

Segment labels: the model label is the model file's slugged stem; the config
label is `{backend}-dt{dt}`; the params label is `base` when no per-cell
overrides apply, otherwise the sorted `name=value` pairs joined by `_`
(sanitized), falling back to the literal `draws` when that would exceed 96
characters; the scenario label is the preset name or `baseline`; the seed label
is `seed_{N}` with the base seed verbatim, unpadded.

Because a level hash is a structural digest of typed values rather than of a
string, the empty scenario is a **constant**: a scenario with no overrides, no
enables and no disables always hashes to the same value, so `baseline-33233b72`
identifies the unmodified baseline in every project. `seed_1-06cbd6b3` and
`base-...` are constants in the same way.

A **multi-cell** `simulate` (`--replicates`, `--seeds`, several `--scenario`, or
`--draws`) writes the N per-cell `sim` leaves _and_ one `sim_ensemble` leaf
under `ensembles/` holding the combined wide-format `ensemble.tsv` that
interleaves every cell with `replicate` / `scenario` / `draw` columns. The
ensemble's `grid` level digests the sorted cell list — each cell contributing
`(scenario_label, process_seed, draw_idx, sim_run_id)` — plus the explicit cell
count, so three replicates and four replicates are different ensembles
(count-in-the-key), and the ensemble `deps` on the N `sim` leaves it was built
from. Its path label is `cells-n{N}`.

`camdl batch run` writes the same per-cell `sim` leaves and no ensemble, plus
`model.ir.json`, `model.render.json`, and `model.graph.json` at the **output
root** (not under `sims/`).

**The `model` label differs by entry point.** `simulate` labels the model level
with the `.camdl` source stem; `batch run` labels it with the stem of the
_compiled IR path_, which is the IR cache's 64-hex filename, or `camdl_{pid}`
when the IR cache is bypassed. The same model, params, scenario and seed
therefore produce the same `run_id` at two (or more) different paths depending
on how the run was launched, and `camdl show <run_id>` reports those paths as an
ambiguous match that no longer prefix can resolve. Until the two entry points
agree on the label, address such a run by its leaf path.

### 2.5 Per-leaf metadata

Every leaf directory contains exactly one `run.json`, the serialized
`RunRecord`. It records the leaf's identity levels, the exact set of files the
leaf owns with their digests, the upstream artifacts it consumed, its completion
status, and a free-form provenance block. It is never itself hashed — identity
comes only from the resolved inputs, which is precisely why attaching an `obs/`
child or a column schema cannot change the parent's `run_id`.

`run.json` also carries an `output_schema` declaring, for each tabular file the
leaf wrote, which column is the time or iteration axis, which are grouping keys,
which are model quantities, and which are sampler diagnostics — so a consumer
can render or join a TSV without reverse-engineering its header.

The full field-by-field contract is §9.5; the column-role vocabulary and the
rendering rule it implies are §10.8.

**Sub-artifacts have no `run.json`.** A trajectory leaf may declare an
observation ensemble as a child at `obs/{obs_hash8}-{obs_seed}/`, holding one
`<stream>.tsv` per observation stream plus an `obs.json` provenance file. That
child is keyed on the model's observation blocks alone, so changing a reporting
parameter re-samples the observations without invalidating the cached
trajectory. Reach it through the parent's `children` map, not by walking the
tree for `run.json` files.

### 2.6 Store-root files

`<root>/index.json` is a rebuildable derived index mapping each `run_id` to its
path and summary fields, so prefix resolution for `show` and `cat` does not
re-walk the whole tree on every call. It is a cache and never authoritative;
§9.6 gives the two invariants that make that operational rather than
aspirational.

`<root>/.staging/` holds in-flight atomic commits (§2.3.1). Entries left behind
by a crash are inert debris; they are never read.

`camdl batch run` additionally archives `model.ir.json`, `model.render.json`,
and `model.graph.json` at the store root.

---
## 3. Core Types

A run is a point in three axes — **M** (parameter values), **σ** (scenario), and
**S** (seed) — and the job of this section is to say which type owns which axis,
where a value stops being a request and becomes a resolved fact, and how that
resolved fact becomes a directory on disk.

Three type families do the work, and conflating them is the usual source of
confusion:

| Family              | Purpose                                                | Lives in                                                      |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------------------- |
| **Surface types**   | Parse one front end: clap args or a TOML document       | `rust/crates/cli/src/args/`, `rust/crates/cli/src/batch.rs`     |
| **Engine input**    | The resolved job both front ends converge on            | `rust/crates/cli/src/sim_job.rs`                                |
| **Identity values** | Hashed, per-level digests of the fully-resolved inputs  | `rust/crates/runid/src/inputs.rs`                               |

The pipeline is:

```text
camdl simulate <args>  ─┐
                        ├─→  SimulateJob  ─→  engine::run_job  ─→  per-cell SimRun
camdl batch run <toml> ─┘     (resolved)        (cell grid)          (resolved θ, σ, seed)
                                                                            │
                                          runid identity values  ←──────────┘
                                          (ModelDigest, SimConfig, ResolvedParams,
                                           ResolvedScenario, Seed)
                                                    │
                                        five LevelIds  →  run_id  →  store path
```

The surface types are deliberately **not** shared between the CLI and the batch
TOML, and `SimulateJob` deliberately has no serde derives. The property that
matters is that both front ends *converge on the same resolved struct*, not that
they share a wire format — a shared serde representation would create a second,
drifting schema alongside batch's existing TOML.

### 3.1 SimulateJob — the resolved engine input

`rust/crates/cli/src/sim_job.rs`

```rust
#[derive(Debug, Clone)]
pub struct SimulateJob {
    /// Resolved IR/`.camdl` path (already anchored to the config dir for
    /// batch; verbatim for the CLI).
    pub model: String,
    /// `--params` / `[config].params` files, applied in order.
    pub params_files: Vec<String>,
    pub backend: ForwardBackend,
    pub dt: f64,
    /// Optional CLI `--integrator` override (method only); `None` → the
    /// model's declared integrator.
    pub integrator: Option<crate::args::types::IntegratorArg>,
    /// Where parameter vectors come from (the central dispatch).
    pub source: ParamSource,
    /// σ layer — which scenarios to run. Empty ⇒ a single implicit baseline.
    pub scenarios: Vec<ScenarioRef>,
    /// S layer.
    pub seeds: Seeds,
    /// `--param NAME=VALUE` CLI overrides merged on top of every cell
    /// (M layer, highest precedence). Empty for batch.
    pub cli_overrides: Vec<(String, f64)>,
    /// `--param-vec PREFIX=FILE` entries (CLI only).
    pub set_vec_entries: Vec<(String, String)>,
    /// `--table NAME=FILE` entries (CLI only).
    pub table_files: Vec<(String, String)>,
    /// Synthetic-observation output mode.
    pub obs: ObsOutput,
    /// Rayon thread count for the simulation phase (1 = sequential).
    pub parallel: usize,
}
```

It carries no output directory, no `--force`, no label, and no GeoJSON path:
those are write-side concerns owned by the sink, not by the job. `camdl
simulate` builds one in `main.rs` and always sets `parallel: 1` (its combined
wide-format output is order-sensitive); `camdl batch run` builds one in
`batch.rs` and passes the manifest's thread count.

The engine that consumes it is `rust/crates/cli/src/engine.rs`:

```rust
pub fn plan_grid(job: &SimulateJob) -> (Vec<CellSpec>, Grid);
pub fn run_job(job: &SimulateJob, sink: &mut dyn RunSink) -> Result<(), String>;
```

`plan_grid` expands the job into an ordered list of cells — the Cartesian
product `scenario × param-point × replicate`, in exactly that iteration order —
and `run_job` runs them (sequentially, or via Rayon when `parallel > 1`) and
merges each into a `RunSink` in canonical order. `simulate` supplies a sink that
accumulates a combined wide-format TSV; `batch run` supplies one that commits a
content-addressed leaf per cell. The two produce byte-identical leaves for the
same cell.

Per-cell seeds are order-independent by construction, so parallelism never
perturbs a trajectory:

```text
process_seed = if explicit --seeds { seeds[rep] }
               else if total_runs == 1 { base_seed }
               else { base_seed ^ (point_idx * SEED_MIX_DRAW)
                                ^ (rep       * SEED_MIX_REP) }
obs_seed     = process_seed ^ SEED_MIX_OBS
```

The scenario index is deliberately **absent** from the mix: paired scenarios at
the same (point, replicate) share a seed, which is what makes their
pre-divergence trajectories byte-identical (common random numbers). The mix
itself lives once, in `util::mix_cell_seed`; `tests/determinism_pin.rs` pins it.

### 3.1.1 ObsOutput — synthetic observation output

`rust/crates/cli/src/sim_job.rs`

```rust
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ObsOutput {
    #[default]
    None,
    /// Single wide-format TSV (errors if streams have different schedules).
    File(PathBuf),
    /// One TSV per stream in a directory.
    Dir(PathBuf),
    /// Like `File`, trajectory suppressed.
    OnlyFile(PathBuf),
    /// Like `Dir`, trajectory suppressed.
    OnlyDir(PathBuf),
}
```

Three accessors keep the interpretation in one place:
`suppresses_trajectory()`, `file_path()`, `dir_path()`.

CLI mapping:

```text
--obs cases.tsv          → ObsOutput::File("cases.tsv")
--obs-dir obs/           → ObsOutput::Dir("obs/")
--obs-only cases.tsv     → ObsOutput::OnlyFile("cases.tsv")
--obs-only-dir obs/      → ObsOutput::OnlyDir("obs/")
```

These four flags are mutually exclusive at the clap level. All of them are
*mirrors*: the observation draws are also written into the run's leaf under
`obs/{obs_hash8}-{obs_seed}/`. In a batch manifest the equivalent switch is
`[obs] enabled = true`, which writes only the leaf copy.

### 3.2 ParamSource — where parameter vectors come from

`rust/crates/cli/src/sim_job.rs`

Exactly one variant is active per job. It answers "how many parameter points is
this job, and where did each one's values come from?" — not "how were they
specified"; the recipe (a `linspace`, a `--draws prior -n 500`) is resolved to
concrete rows by the front end before the job is built.

```rust
#[derive(Debug, Clone)]
pub enum ParamSource {
    /// Single point: base params + CLI overrides, run `replicates` times.
    Point { replicates: usize },
    /// Deterministic grid: Cartesian product of swept values.
    Sweep { points: Vec<IndexMap<String, f64>>, replicates: usize },
    /// Pre-resolved parameter draws (posterior file / prior / uniform).
    Draws {
        rows: Vec<IndexMap<String, f64>>,
        replicates: usize,
        /// `Some(path)` iff the draws came from a user-authored file
        /// (`--draws <file.tsv>`); `None` for generated draws.
        explicit_file: Option<PathBuf>,
    },
}
```

Two accessors: `param_points()` returns the ordered per-cell override maps (a
single empty map for `Point`), and `replicates()` returns the replicate count.
Total cells are `|scenarios| × |param_points| × replicates`, except that an
explicit seed list overrides the replicate count — see §3.5.

Which front end produces which variant:

| Variant  | Produced by                                                              |
| -------- | ------------------------------------------------------------------------ |
| `Point`  | `simulate` with no `--draws` (`--replicates N` sets the count)            |
| `Sweep`  | `batch run` — the manifest's `[sweep]` grid, or a `[design.*]` block       |
| `Draws`  | `simulate --draws …` (all four sources of §3.4)                           |

`explicit_file` is what distinguishes a user-authored draws file from generated
draws, and the distinction has teeth: if a scenario sets a parameter that a
`--draws <file.tsv>` also provides as a column, that is a hard error naming the
parameter, the scenario, and the file — the user pinned θ two ways and the
intent is ambiguous. For generated draws the scenario simply wins. The check is
`engine::check_explicit_draws_scenario_collision`.

Note that a sweep point loses to a scenario `set`/`scale` for the same reason a
generated draw does (see the resolution order in §1.3): both are *automated*
M-layer variation, whereas a scenario is a deliberate counterfactual and
`--param` is the user's explicit assertion about this run.

### 3.3 Sweep specification

There are two sweep surfaces, and they are different types with different
syntax. Neither is reachable from `camdl simulate`.

**Batch manifest `[sweep]`** — `rust/crates/cli/src/batch.rs` (private to the
module). One entry per swept parameter; the section expands to the Cartesian
product, with keys sorted for a deterministic point order.

```toml
[sweep]
vacc_eff = [0.1, 0.3, 0.5]                              # explicit list
beta     = { linspace = { min = 0.1, max = 0.9, n = 9 } }
kappa    = { logspace = { min = 0.001, max = 0.1, n = 5 } }
R0       = { range = { min = 1.0, max = 5.0, step = 0.5 } }
```

`linspace` and `logspace` are endpoint-inclusive and collapse to `[min]` at
`n = 1`; `logspace` interpolates geometrically between `min` and `max` (the
values, not their exponents). `range` walks `min` upward by `step` while
`x <= max`, with `step` defaulting to `1.0`.

**CLI `--sweep NAME=SPEC`** — `rust/crates/cli/src/args/types.rs`, used by
`camdl profile`, `camdl fit run`, and `camdl fit predict`.

```rust
pub struct SweepSpec { pub name: String, pub grid: Grid }

pub enum Grid {
    List(Vec<f64>),                          // beta=0.1,0.2,0.3
    Linear { min: f64, max: f64, n: usize }, // beta=lin(1.0,4.0,11)
    Log10  { min: f64, max: f64, n: usize }, // k=log10(0.001,1000,7)
}
```

`lin` and `log10` require `n >= 2` and `min < max`; `log10` additionally
requires `min > 0`. The function is spelled `log10`, not `log`, because the
camdl DSL binds `log` to the natural logarithm inside rate expressions — writing
`log(...)` here is a hard error with a redirect rather than a silent
reinterpretation.

### 3.4 Parameter draws

Draws are parameter vectors sampled from a distribution — a different object
from a designed sweep grid. A sweep is a design you chose; draws are samples
from inference output or a prior, and downstream analyses need to know which
they are looking at.

There is no `DrawsSpec` type. `simulate --draws` resolves its argument to
concrete rows *before* the job exists, and the rows enter as
`ParamSource::Draws`.

| `--draws <value>` | Source                                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------ |
| `<file.tsv>`      | A TSV of parameter vectors (one row per draw, columns = parameter names).                                     |
| `prior`           | Sample declared priors. Requires `-n N`. `--fit <fit.toml>` is optional.                                      |
| `uniform`         | Sample uniformly from each parameter's `in [lo, hi]` bounds. Requires `-n N`.                                 |
| `posterior`       | A completed fit's canonical post-warm-up `draws.tsv`. Requires `--fit <fit results dir>`.                     |

`--draws prior` resolves priors through the same three-tier precedence chain
`camdl fit run` uses — fit.toml `[estimate].prior` first, then the model IR's
`~ <dist>` declarations, then a flat fallback. With no `--fit` it samples the
model's own declared priors; a parameter with no usable prior in either tier is
a hard error naming the remediation. Do **not** describe this source as
requiring a fit config.

`--draws uniform` samples `lo + (hi - lo)·U` per parameter from the model's
bounds, falling back to a parameter's resolved default when it declares no
bounds, and erroring when it has neither. This is space-filling exploration for
model debugging, not a prior.

`--draws posterior --fit <fit results dir>` reads the draws the terminal
Bayesian stage (PGAS / PMMH / MH / NUTS) wrote — `<stage_dir>/draws.tsv`, which
is post-warm-up and thinned and carries every model parameter. Resolution is by
artifact, not by method name: a stage has a posterior iff it wrote a
`draws.tsv`, so an optimizer-only fit (IF2 / NLopt) resolves to an error rather
than to a single point dressed up as a distribution
(`rust/crates/cli/src/posterior_draws.rs`).

`--draws <file.tsv> --fit <config-or-run>` additionally **backfills** any
parameter absent from the file's columns from the fit's `[fixed]` block, never
overwriting a column the file provides — a raw posterior trace tail carries only
the estimated columns, so this fills the fixed parameters from the fit rather
than falling back to model defaults.

`--draws-out PATH` writes the resolved rows back out in the same
one-row-per-draw, one-column-per-parameter format `--draws PATH` reads, so a
generated cloud round-trips.

### 3.5 Seeds — the S layer

`rust/crates/cli/src/sim_job.rs`

```rust
#[derive(Debug, Clone)]
pub enum Seeds {
    /// A single base seed; replicates derive seeds via XOR-mixing.
    Single(u64),
    /// An explicit list. Each seed is a seed-slot used verbatim —
    /// "seed N means the same trajectory."
    Explicit(Vec<u64>),
}
```

The load-bearing distinction is whether the seeds were given *explicitly*.
`Seeds::explicit()` returns `Some(&[..])` only for the list form, and the engine
branches on it: an explicit list indexes directly by replicate, so seed `N`
names the same trajectory in every run that mentions it; a single base seed
derives per-cell seeds by XOR-mixing the point and replicate indices (§3.1).
With an explicit list, the list length *is* the replicate count — the
`ParamSource`'s own `replicates` is not consulted.

Surfaces:

| Surface                       | Result                                       |
| ----------------------------- | -------------------------------------------- |
| `simulate --seed 42`          | `Seeds::Single(42)` (default `1`)             |
| `simulate --seeds 1:100`      | `Seeds::Explicit([1..=100])`                  |
| `simulate --seeds 1,2,42`     | `Seeds::Explicit([1, 2, 42])`                 |
| batch `[config] seeds = {…}`  | `Seeds::Explicit(SeedsSection::resolve())`    |

`--seed` and `--seeds` conflict, as do `--seeds` and `--replicates`. The batch
manifest's seed table lives under `[config]` and is a table, never a bare
integer (`rust/crates/cli/src/batch.rs`, `SeedsSection`):

```toml
[config]
seeds = { list = [42, 137, 256] }   # explicit list
seeds = { n = 1000 }                # start..start+n, start defaults to 1
seeds = { n = 100, start = 500 }    # 500..600
seeds = { from = 1, to = 1000 }     # inclusive range
# omitted entirely                  → [1]
```

### 3.6 ScenarioRef — the σ layer

`rust/crates/cli/src/sim_job.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScenarioRef {
    /// Reference a scenario defined in the `.camdl` file.
    Named(String),
    /// Inline definition (a `[[scenario]]` entry carrying patches).
    Inline {
        name: String,
        #[serde(default)] enable: Vec<String>,
        #[serde(default)] disable: Vec<String>,
        #[serde(default)] params: IndexMap<String, f64>,
    },
}
```

A reference is resolved against the model's `scenarios{}` presets before it can
be run, and resolution is total — there is no "maybe it is a preset" state
downstream:

```rust
pub enum ResolvedScenario {
    /// The name matched a model preset; the preset is the source of truth
    /// for enable/disable/set/scale/compose.
    Preset { name: String },
    /// The name matched nothing but inline patches were given.
    Adhoc { name: String, enable: Vec<String>, disable: Vec<String>,
            params: Vec<(String, f64)> },
}

pub fn resolve_scenario_ref(scenario: &ScenarioRef, model_preset_names: &[String])
    -> Result<ResolvedScenario, String>;
```

The four cases:

1. Name matches a preset, no inline fields → `Preset`.
2. Name matches nothing, inline fields present → `Adhoc`.
3. Name matches nothing, no inline fields → **error**, listing the available
   presets. Exception: the sentinel names `baseline` (`simulate`) and `fitted`
   (`camdl fit predict`) always resolve to the identity patch, even on a model
   that declares no `scenarios{}` block at all.
4. Name matches a preset **and** carries inline fields → **error**. A scenario
   reference is either a model preset or an ad-hoc patch, never both; the model
   scenario is the source of truth.

When no `[[scenario]]` entries are defined and no `--scenario` flag is given, a
single implicit baseline is used. This is not a "default scenario" — it is the
absence of any scenario patch, and it hashes to the real digest of the empty
delta rather than to a literal zero.

Because a preset resolves at a higher precedence tier than a sweep point or a
generated draw (§1.3), a preset whose `set` block mentions a swept parameter
silently wins over the sweep. Choose preset names and sweep axes so they do not
overlap.

### 3.7 From a job to a path: run identity

Output paths are never constructed ad hoc. A command resolves its inputs into
typed identity values, hashes them into per-level digests, and the store derives
the path from those digests. Two properties follow: the same inputs always land
in the same directory (so a re-run is a cache hit), and different inputs never
share a directory (so a stale result cannot be served for changed inputs).

**Step 1 — resolve.** `rust/crates/cli/src/resolve.rs` maps a cell's already-
resolved CLI inputs into `runid` identity values. Every field entering a hash is
a concrete value: parameter and scenario maps are name→value, the seed is the
resolved `process_seed` (never the base `--seed`), and a non-finite resolved
float is a `ResolveError` raised before any hashing.

```rust
pub fn resolve_trajectory(ctx: &TrajectoryCtx) -> Result<ResolvedTrajectory, ResolveError>;

pub struct ResolvedTrajectory {
    pub levels: Vec<LevelId>,     // five levels, in path order
    pub run_id: ContentHash,      // derived from the level hashes
}
```

**Step 2 — the five levels.** A forward simulation's identity is factored into
five digests, each hashing a disjoint slice of the input set
(`rust/crates/runid/src/inputs.rs`):

| Level      | Identity value      | What it folds in                                                            |
| ---------- | ------------------- | ---------------------------------------------------------------------------- |
| `model`    | `ModelDigest`       | The whole canonical IR, plus `ir_version` and the engine version              |
| `config`   | `SimConfig`         | backend, `dt`, `t_start`, `t_end`, output schedule, calendar mode, `allow_degenerate_rates`, `no_flows`, `columns` |
| `params`   | `ResolvedParams`    | The resolved base parameter map, plus content digests of any `--table` files  |
| `scenario` | `ResolvedScenario`  | Sorted enabled/disabled intervention ids and the canonical parameter patch    |
| `seed`     | `Seed`              | `process_seed` only                                                           |

Two rules govern membership, and they are the whole discipline:

- **A field that changes the stored bytes is identity and must re-key.** This
  includes output-view flags: `--no-flows` and `--columns` change which columns
  the leaf contains, so they ride in `SimConfig`. `--output-every` is not there
  because it lowers into the model's output schedule and therefore rides the
  model digest instead.
- **A re-encoding of the same values is presentation and is stripped.**
  `output.format` and `simulation.time_semantics` are blanked by
  `inputs::normalize_for_hash` inside `ModelDigest::from_model` — the one
  constructor every identity path uses, so no artifact kind can silently opt
  out.

Fields marked `#[run_input(provenance)]` are recorded but never hashed:
`Seed::base_seed` and `ArtifactRef::kind` are the two on the simulate path.
Everything else in a `#[derive(RunInput)]` type is folded, in declaration order,
with no skip-if-default — adding a field re-keys every existing leaf of that
kind, which is deliberate versioned turnover, scoped by bumping either that
type's `#[run_input(schema_version = N)]`, the crate-wide `HASH_VERSION`, or
`ir/VERSION`.

**Step 3 — the leaf address.** `runid::run_id` folds the artifact kind's
declaration index and the count-prefixed level-hash sequence
(`rust/crates/runid/src/kind.rs`), so two kinds with a coincidentally equal level
sequence cannot alias.

```rust
pub fn run_id(kind: ArtifactKind, levels: &[ContentHash]) -> ContentHash;
```

**Step 4 — the path.** `runid::store_path` (`rust/crates/runid/src/layout.rs`)
renders one `{label}-{hash8}` segment per level under the kind's partition
directory. The label is provenance — a rename yields a new directory and a
harmless cache miss — and the 8 hex characters are identity. Navigation and
display read `run.json`, never the segments.

```rust
pub fn store_path(root: &Path, kind: ArtifactKind, levels: &[LevelId]) -> PathBuf;
```

The output root resolves as **CLI `--output-dir` > config-file `output_dir` >
`CAMDL_OUTPUT_DIR` > `results/`** (`rust/crates/cli/src/run_paths.rs`,
`output_root`). Project-specific config sits above ambient shell state
deliberately: setting the environment variable must not silently redirect a fit
that declared its own `output_dir`.

A real single-run leaf:

```text
results/sims/
  sir_basic-cd37d79d/                 # model:    IR digest + ir_version + engine
    chain_binomial-dt1-9fe90ef5/      # config:   backend, dt, schedule, output view
      base-b912ced1/                  # params:   resolved base values (+ table digests)
        baseline-33233b72/            # scenario: the (here empty) delta
          seed_7-d0aef62a/            # seed:     the resolved process_seed
            traj.tsv
            run.json
```

The `params` label is `base` for an unswept run and a key-sorted `k=v` join for
a sweep point or draw row, collapsing to the tag `draws` when the join would
overflow a path component.

**Step 5 — write.** Every CAS write goes through one seam,
`resolve::begin_resolved_write`, which derives the path from the resolved
identity, builds the `RunRecord` in exactly one place, and dispatches to either
an atomic commit (the whole artifact set handed over at once — the simulate and
batch path) or a streaming claim that the caller writes into and finalizes (the
fit path). Cache validity is `FsCasStore::lookup`, which requires an identity
match, a current schema, `status = Completed`, and an exact-set file check;
anything else is a `Miss`, a `Stale(reason)`, or a `Collision` on a
short-hash-prefix clash, which is disambiguated rather than overwritten.

**Derived ensembles.** A multi-cell `simulate` also writes the combined
wide-format TSV as its own artifact kind, `SimEnsemble`, under
`results/ensembles/` with four levels — `model` / `config` / `params` / `grid`
(`rust/crates/cli/src/sim_ensemble_cas.rs`). The `grid` level folds the sorted
per-cell list *and the explicit cell count*, so three replicates and four
replicates are different ensembles; the ensemble's `deps` carry one
`ArtifactRef` per cell leaf, pinned to that cell's `traj.tsv` digest.
## 4. Single-Run Simulation (No Batch File)

The `.camdl` file plus a parameter TOML is a complete, self-contained
specification for running a model. No batch file or fit config is needed for
exploration.

### 4.1 Basic CLI

```bash
# Baseline (no scenario) — writes a leaf in the content-addressed store
# and prints where it landed on stderr (§4.4)
camdl simulate seir_vaccine.camdl --params params.toml --seed 42
```

```
   stored ./results/sims/seir_vaccine-a0c10f3e/chain_binomial-dt1-e669f2d2/base-eb8ec4ef/baseline-33233b72/seed_42-dd2fb524
          camdl cat 18bd7a09b098c20c22fd478abd331b5475d213c57f5505ead75088f35a523c27
```

The four hex suffixes and the `run_id` are content hashes; they change whenever
any identity-bearing input changes (including the runtime version), so the
exact digests in this document will not match your run.

```bash
# Named scenario from the model's `scenarios {}` block
camdl simulate seir_vaccine.camdl --params params.toml --scenario with_sia --seed 42

# Ad-hoc intervention toggle (no named scenario)
camdl simulate seir_vaccine.camdl --params params.toml --enable sia_round_1 --seed 42

# Parameter override (M layer)
camdl simulate seir_vaccine.camdl --params params.toml --param beta=0.5 --seed 42

# Scenario + parameter override (σ layer + M layer — both valid)
camdl simulate seir_vaccine.camdl --params params.toml --scenario with_sia --param beta=0.5 --seed 42

# Resolve everything and print the plan without simulating
camdl simulate seir_vaccine.camdl --params params.toml --seed 42 --dry-run
```

`--dry-run` prints the resolved plan and each parameter's provenance:

```
camdl simulate (dry run)

  model: seir_vaccine.camdl
  backend: chain_binomial
  dt: 1
  seed: 42
  scenario: (baseline)

Parameters (7):
  I0        =             10  params.toml
  N0        =         100000  params.toml
  beta      =       0.300000  params.toml
  gamma     =       0.100000  params.toml
  omega     =       0.003000  params.toml
  sigma     =       0.200000  params.toml
  vacc_frac =       0.800000  params.toml
```

`--seed` defaults to `1` (env `CAMDL_SEED`); a run without an explicit seed is
still deterministic and still content-addressed.

### 4.2 CLI Flag Rules

**`--scenario` and `--enable`/`--disable` are mutually exclusive (both σ
layer).** clap rejects the combination at parse time, before the model is
compiled:

```bash
camdl simulate seir_vaccine.camdl --params params.toml --scenario with_sia --enable sia_round_1
```

```
error: the argument '--scenario <SCENARIOS>' cannot be used with '--enable <ENABLE>'

Usage: camdl simulate --params <FILE> --scenario <SCENARIOS> <MODEL>

For more information, try '--help'.
```

To combine a preset with an extra toggle, define a composed scenario in the
model file.

**`--param` is always valid (M layer, independent of σ layer):**

```bash
camdl simulate seir_vaccine.camdl --params params.toml --param beta=0.5 --seed 42
camdl simulate seir_vaccine.camdl --params params.toml --scenario with_sia --param beta=0.5 --seed 42
```

**Other exclusions enforced at parse time:**

| combination                                        | why                                             |
| -------------------------------------------------- | ----------------------------------------------- |
| `--seed` × `--seeds`                                | one base seed vs an explicit seed list          |
| `--seeds` × `--replicates`                          | explicit seeds vs derived seeds                 |
| `--stdout` × `--seeds` / `--replicates` / `--draws` | `--stdout` is single-cell only                  |
| `--stdout` × `-o` / `--obs*`                        | those mirror the store `--stdout` opts out of   |
| `--event-log` × `--seeds` / `--replicates` / `--draws` | event-log recording is single-run only       |
| `--draws-out`, `--fit` without `--draws`            | both are companions to `--draws`                |

`--obs-only` cannot be combined with `--obs`/`--obs-dir`, and `--obs-only-dir`
cannot be combined with any of the three; those are checked after parsing and
exit 1 with a one-line message.

**Precedence of parameter sources (last wins).** Model defaults → `--params`
files in order → draw row or sweep point → scenario `set`/`scale` (preset or
inline) → `--param`. A scenario that `set`s a parameter therefore overrides a
value coming from `--draws` or from a batch `[sweep]`; only `--param` beats a
scenario. See §5.4.

### 4.3 Parameter and Model Inputs

```bash
--params FILE              # load a parameter TOML (may be repeated; applied in order)
--param NAME=VALUE         # override a single parameter (may be repeated)
--param-vec PREFIX=FILE    # override an indexed parameter family from a file
--table NAME=FILE          # supply the data for a table declared `external("NAME")`
```

A model declaring `C : patch × patch = external("contact")` requires the
matching `--table`:

```bash
camdl simulate sir_ext_table.camdl --scenario baseline --seed 1
```

```
error: table 'contact' is declared as external() but --table contact=<file> was not provided
```

```bash
camdl simulate sir_ext_table.camdl --scenario baseline --seed 1 --table contact=contact.tsv
```

The table file is a headerless numeric TSV whose shape matches the declared
dimensions; the file's bytes enter the run identity, so editing it produces a
new leaf.

**Backend and integration knobs.**

```bash
--backend {gillespie|chain_binomial|ode}   # default chain_binomial
--dt N                                     # step size for discrete-time backends (default 1.0)
--integrator {rk4|rk45}                    # ODE method override; tolerances live in the model
```

**Trajectory column view.** These change the emitted columns *and* the run
identity, so a filtered run is a distinct leaf:

```bash
--output-every N     # one row every N time-units, overriding the model's `output { every }`
--no-flows           # drop every `flow_*` column
--columns S,I        # allow-list of output columns; emitted order follows the model
--dates              # add a calendar `date` column (requires the model to declare `origin`)
```

```bash
camdl simulate seir_vaccine.camdl --params params.toml --seed 42 \
    --output-every 30 --columns S,I --stdout
```

```
# 0.1.0+3e2b2888 (2026-08-10)
t	S	I
0	99990	10
30	99791	64
60	96434	1208
```

Without an `origin` in the model, `--dates` is an error:

```
error: --dates requires the model to declare an `origin` (e.g. `origin = date("2020-01-01")`).
```

### 4.4 Output destinations — the content-addressed store by default

Every run writes a leaf into the content-addressed store under `--output-dir`
(default `./results`, env `CAMDL_OUTPUT_DIR`) and reports the leaf path plus the
`camdl cat` that reads it back. Both lines go to **stderr**, so a `--stdout`
pipeline stays clean. A `--cas` flag is accepted and ignored — content
addressing is not opt-in.

```bash
# Default: write the leaf, print where it went.
camdl simulate seir_vaccine.camdl --params params.toml --seed 42
#   stored ./results/sims/…/seed_42-dd2fb524
#          camdl cat 18bd7a09…

# Opt out: stream the trajectory to stdout, write no leaf.
camdl simulate seir_vaccine.camdl --params params.toml --seed 42 --stdout > out.tsv
camdl simulate seir_vaccine.camdl --params params.toml --seed 42 --stdout | head

# Write a plain-TSV mirror AND the leaf (the file is a convenience copy).
camdl simulate seir_vaccine.camdl --params params.toml --seed 42 -o out.tsv
#   trajectory written to out.tsv
#   stored ./results/sims/…

# Browse what you have run.
camdl list
camdl show <run-id-prefix>
camdl cat  <run-id-prefix>
```

**The three destinations.** A run always produces exactly one primary
trajectory artifact; these say where it goes.

| flag                        | trajectory                       | store leaf written? |
| --------------------------- | -------------------------------- | ------------------- |
| _(default)_                 | store leaf only                  | yes                 |
| `-o FILE` / `--output FILE` | store leaf **and** `FILE`        | yes                 |
| `--stdout`                  | stdout only                      | no                  |

The trajectory is **never** written to stdout without `--stdout`, at any size.
Commands whose primary result is a scalar do echo it: `camdl pfilter` prints the
log-likelihood to stdout while also writing its own leaf, so the tight
"vary θ, re-check" loop stays a one-liner:

```bash
camdl pfilter seir_observations.camdl --scenario baseline \
    --data weekly_cases=obs/weekly_cases.tsv --data detection=obs/detection.tsv \
    --particles 200 --seed 1 2>/dev/null
```

```
-190.8639
```

**Mirror files carry a provenance header.** `--stdout` and `-o` both prefix the
TSV with a comment line naming the binary that produced it:

```
# 0.1.0+3e2b2888 (2026-08-10)
t	S	E	I	R	V	flow_infection	flow_progression	flow_recovery	flow_waning
```

The bytes in the store leaf's `traj.tsv` are the same table without that header.

**Re-running does not skip.** `camdl simulate` re-simulates every cell on every
invocation and re-commits the identical leaf; the commit is idempotent, so the
store does not grow, but the CPU cost is paid again. There is no cache-hit
short-circuit and no "cached" banner on this path, and `--force` therefore has
no observable effect on `simulate`. `camdl batch run` **does** skip committed
leaves (§5.6), and `--force` is meaningful there.

**Layout.** Five factored levels under `<output-dir>/sims/`:

```
sims/{model}-{h8}/{config}-{h8}/{params}-{h8}/{scenario}-{h8}/seed_{n}-{h8}/
    run.json      # identity, levels, artifact digests, output schema, provenance
    traj.tsv      # the trajectory
    obs/          # present only when synthetic observations were requested
```

Each component is `<human label>-<first 8 hex of that level's hash>`. The label
is display only; identity is the hash. A params level with no overrides is
labelled `base`; a full drawn parameter vector collapses to `draws` so the path
component stays under `NAME_MAX`.

**Hash composition.** The `model` level keys on the compiled IR; `config` on
backend, `dt`, integrator, and the column view; `params` on the resolved base
parameters, draw/sweep overrides, and the bytes of any `--table` file;
`scenario` on the resolved enable/disable/set delta; `seed` on the process seed.
The runtime version participates, so a code change that alters simulation
semantics produces new leaves rather than silently reusing old ones.

`--no-flows` / `--columns` re-key the `config` level only. `--output-every`
rewrites the compiled IR's output schedule, so it re-keys the `model` level as
well — same model source, different model hash. `--emit-every` re-keys neither:
it touches only emitted observations, so it keys the leaf's `obs/` child instead
(§10.5).

**Ensembles.** `--seeds`, `--replicates`, and `--draws` write one leaf per cell
plus a single combined `SimEnsemble` artifact under `<output-dir>/ensembles/`:

```bash
camdl simulate seir_vaccine.camdl --params params.toml --scenario baseline,with_sia --seeds 1:3
```

```
2 scenarios × 3 replicates = 6 runs
  storing ensemble · 6 cells (276.2KB)
 ensemble ./results/ensembles/seir_vaccine-a0c10f3e/chain_binomial-dt1-72322740/base-2c2d41e8/cells-n6-70131f1a
   stored 6 leaves · ./results/sims/
          camdl list
```

`camdl cat` on the ensemble emits the combined wide-format TSV, with leading
index columns gated on the grid shape — `replicate` always, then `scenario` when
more than one scenario ran, then `draw` under `--draws`:

```
# 0.1.0+3e2b2888 (2026-08-10)
replicate	scenario	t	S	E	I	R	V	flow_infection	flow_progression	flow_recovery	flow_waning
1	baseline	0	99990	0	10	0	0	0	0	0	0
```

`replicate` is a 1-based cell index, not the seed. With `--seeds 10,20` the
column still reads `1` and `2`; the seed that produced a row is recorded in the
per-cell leaf's `run.json`, not in the combined TSV.

`--parallel` is accepted by `camdl simulate` but not honoured — the simulate
grid always runs sequentially. Use `camdl batch run --parallel N` for a
parallel grid.

### 4.5 `camdl list` / `show` / `cat` / `label` — the primary access path

Because output goes to the store by default, these are **the** way you reach
results. `list` defaults to `./results` (positional `ROOT`, `--root DIR`, or
`CAMDL_OUTPUT_DIR`); every run's banner hands you the exact `camdl cat` to run
next.

```bash
# Tabular overview, most recent first, one section per artifact kind
camdl list
camdl list --since 1h              # last hour
camdl list --model seir_vaccine    # substring match on the model path
camdl list --scenario baseline --all
camdl list --kind sim              # sim | fit | profile | pfilter | survey | ensemble | all
camdl list --format json           # machine-readable; see the caveat below

# Full metadata for one run (short-hash prefix resolves git-style)
camdl show 18bd7a09
camdl show ./results/sims/seir_vaccine-a0c10f3e/chain_binomial-dt1-e669f2d2/base-eb8ec4ef/baseline-33233b72/seed_42-dd2fb524

# Emit the trajectory, or a named stream from the leaf
camdl cat 18bd7a09
camdl cat b1fb4eb0 --stream cases_urban
camdl cat <id> --stream event_log.tsv

# Attach a display label to any run
camdl label 18bd7a09 "reference run"
```

`camdl list` groups by artifact kind and shows:

```
sims
 CREATED   RUN_ID    LABEL          MODEL         SCENARIO  SEED  PARAMS  SIZE  PATH
 just now  52f1a1ba  <unlabelled>   seir_vaccine  baseline  42    base    35K   ./results/sims/seir_vaccine-a0c10f3e/…/seed_42-dd2fb524
 13m ago   18bd7a09  reference run  seir_vaccine  baseline  42    base    32K   ./results/sims/seir_vaccine-a0c10f3e/…/seed_42-dd2fb524
```

with a separate `ensembles` block carrying `CELLS` instead of `SCENARIO`/`SEED`.
`--limit` defaults to 50; `--all` removes the cap.

Paths are printed relative to the root you asked for, so they are copy-paste
ready into `camdl show` / `camdl cat`. Short-hash prefix resolution is git-style:

```bash
camdl show 1
```

```
error: '1' is ambiguous, matches 2 entries:
  sim            ./results/sims/sir_priors-39588ff2/…/seed_5871781006564002450-1277c5c6
  sim            ./results/sims/seir_vaccine-a0c10f3e/…/seed_42-dd2fb524
refine by appending /<scenario> and/or /<seed_N>, or pass a longer hash prefix
```

`camdl show` prints the resolved path, kind, model, scenario, seed, config, the
full `run_id`, every level with its hash, artifact sizes, creation time, engine
version, and the `argv` that produced the run.

`--format json` emits the sim, fit, profile, pfilter, and ensemble sections as
JSON Lines (one object per line) and the survey section as a pretty-printed JSON
array. The stream is therefore not a single JSON document; parse it per section
or filter with `--kind`.

`<root>/index.json` is a derived run_id → leaf index, rebuilt lazily on a
prefix-resolution miss and by `camdl dev reindex <root>`. Deleting it is safe.

### 4.6 Synthetic Observations

```bash
# All streams into one wide TSV (single-cadence models only)
camdl simulate sir_two_patch_long_obs.camdl --scenario baseline --seed 42 --obs cases.tsv

# One TSV per observation stream (multi-stream / mixed-cadence models)
camdl simulate seir_observations.camdl --scenario baseline --seed 42 --obs-dir obs/

# Emit weekly instead of the model's declared cadence — every stream, then one
# stream by its observation-block label (gh#656, §10.5).
camdl simulate seir_observations.camdl --seed 42 --emit-every 7 --obs-dir weekly/
camdl simulate seir_observations.camdl --seed 42 --emit-every cases=7 --obs-dir mixed/

# Independent replicates
camdl simulate sir_two_patch_long_obs.camdl --scenario baseline --seed 42 \
    --replicates 5 --obs cases_rep.tsv

# Suppress the loose trajectory mirror, emit observations only (SBC workflows)
camdl simulate seir_observations.camdl --scenario baseline --seed 42 --obs-only-dir obs_only/
```

`--obs` writes one wide table with a column per stream:

```
time	cases_urban	cases_rural
0	0	0
7	2	3
14	1	0
```

Under `--replicates`/`--seeds`/`--draws` the table gains a leading `replicate`
column:

```
replicate	time	cases_urban	cases_rural
1	0	0	0
```

`--obs` refuses a model whose streams emit on different schedules, before
simulating:

```
error: observation streams have different schedules (weekly_cases: …step: 7.0…, detection: …step: 14.0…).
A single wide TSV cannot hold multi-cadence streams.
Use --obs-dir (one file per stream, keeps trajectory) or
--obs-only-dir (one file per stream, suppresses trajectory).
```

`--obs-only` / `--obs-only-dir` suppress only the *loose* trajectory mirror; the
store leaf still holds `traj.tsv`. In every mode the sampled observations are
also written into the leaf under `obs/<obs-hash>-<obs-seed>/<stream>.tsv`
alongside an `obs.json`, so they are reachable with `camdl cat <id> --stream
<name>` without keeping the loose files.

The observation RNG is seeded independently of the process RNG, so adding
`--obs` does not change the trajectory.
---

## 5. Ensembles and Batch Simulation

Two entry points share one engine. `camdl simulate` covers a grid expressed
entirely in flags (scenarios × draws/replicates × seeds); `camdl batch run`
covers a grid declared in a TOML manifest, which additionally supports Cartesian
`[sweep]` grids and space-filling `[design.*]` blocks. Both build a
`SimulateJob` and hand it to `engine::run_job`, and both write the same
five-level leaves, so `camdl list` / `show` / `cat` browse them uniformly.

There is no `--sweep` flag on `camdl simulate` and no `--batch` flag; a sweep is
a batch manifest.

### 5.1 CLI Invocations

```bash
# ── Multiple scenarios × seeds ───────────────────────────
camdl simulate seir_vaccine.camdl --params params.toml \
    --scenario baseline,with_sia --seeds 1:1000

# --scenario may also be repeated instead of comma-separated:
camdl simulate seir_vaccine.camdl --params params.toml \
    --scenario baseline --scenario with_sia --seeds 1:1000

# ── Stochastic replicates at one parameter point ─────────
camdl simulate seir_vaccine.camdl --params params.toml --replicates 100

# ── Posterior predictive from a draws file ───────────────
camdl simulate sir_priors.camdl \
    --draws draws.tsv --replicates 10 --obs ppc.tsv

# ── Prior predictive (model must declare `~ prior(...)`) ─
camdl simulate sir_priors.camdl --draws prior -n 500 --replicates 5 --obs-dir prior_pred/

# ── Uniform space-filling over declared bounds ───────────
camdl simulate sir_priors.camdl --draws uniform -n 500 --replicates 5 --obs-dir uniform_pred/

# ── Draws from a completed fit's posterior cloud ─────────
camdl simulate sir_priors.camdl --draws posterior --fit results/fits/sir-8a3f12b4 --replicates 10

# ── Record the sampled parameter vectors ─────────────────
camdl simulate sir_priors.camdl --draws uniform -n 5 --draws-out draws_uniform.tsv

# ── From a batch manifest ────────────────────────────────
camdl batch run batches/scenario_comparison.toml --parallel 8
camdl batch run batches/scenario_comparison.toml --dry-run
camdl batch status batches/scenario_comparison.toml
```

`-n` / `--n-draws` sets the draw count for `--draws prior|uniform` (there is no
`--n`). `--fit` is required for `--draws posterior`, optional for
`--draws prior` (a `fit.toml` supplying priors) and for `--draws <file>` (a
`[fixed]` block backfilling columns the file omits).

`--draws prior` on a model with no declared priors names the gap and the fixes:

```
error: parameters 'beta', 'sigma', 'gamma', 'omega', 'vacc_frac', 'N0', 'I0' no prior and no default value.
  Fix options: add `~ prior(...)` to the model, supply `--scenario NAME` if a scenario pins these values,
  supply `--fit FIT.toml`, or use `--draws uniform` for space-filling exploration.
```

A prior whose mass falls partly outside the declared bounds is rejection-sampled
and reported:

```
warning: prior for 'gamma' placed 16.7% mass outside declared bounds (1 rejected / 5 accepted). Consider widening bounds or tightening the prior.
generated 5 prior draws from model IR (7 sampled + 0 fixed params)
```

Combining an explicit draws **file** with a scenario that pins the same
parameters is a hard error — the intent is ambiguous:

```
error: scenario 'baseline' sets parameter(s) [I0, N0, beta, gamma, kappa, rho, take] that the --draws file
'draws_uniform.tsv' also provides as column(s). A draws file pins these parameters per draw, so applying a
scenario that also sets them is ambiguous.
  Fix: drop the column(s) [I0, N0, beta, gamma, kappa, rho, take] from the draws file, or use a scenario that
  does not touch them.
```

The same collision with **generated** draws (`--draws prior|uniform|posterior`)
is not an error: the scenario wins and every draw resolves to the same θ. See
§5.4 — this is the one place where a wrong result looks like a correct one.

### 5.2 CLI ↔ Type Mapping

`camdl simulate` parses into `args::SimulateArgs` (clap) and lowers to
`sim_job::SimulateJob`; `camdl batch run` deserializes `batch::ExperimentToml`
and lowers to the same `SimulateJob`. From there neither the engine nor the
store knows which front-end produced the job.

```rust
/// The resolved engine input. Both front-ends converge here.
pub struct SimulateJob {
    pub model: String,                 // resolved IR or .camdl path
    pub params_files: Vec<String>,     // --params / [config].params, applied in order
    pub backend: ForwardBackend,       // gillespie | chain_binomial | ode
    pub dt: f64,
    pub integrator: Option<IntegratorArg>,   // --integrator rk4|rk45
    pub source: ParamSource,           // where parameter vectors come from
    pub scenarios: Vec<ScenarioRef>,   // σ layer; empty ⇒ one implicit baseline
    pub seeds: Seeds,                  // S layer
    pub cli_overrides: Vec<(String, f64)>,   // --param (highest M tier)
    pub set_vec_entries: Vec<(String, String)>, // --param-vec
    pub table_files: Vec<(String, String)>,  // --table
    pub obs: ObsOutput,
    pub parallel: usize,               // rayon threads; simulate always passes 1
}

pub enum ParamSource {
    /// One point, run `replicates` times. `simulate` with no --draws.
    Point { replicates: usize },
    /// Cartesian grid of swept values. Only `batch run` [sweep]/[design.*].
    Sweep { points: Vec<IndexMap<String, f64>>, replicates: usize },
    /// Pre-resolved draws. `explicit_file` is Some only for --draws <file.tsv>,
    /// which is what gates the scenario-collision error (§5.1).
    Draws {
        rows: Vec<IndexMap<String, f64>>,
        replicates: usize,
        explicit_file: Option<PathBuf>,
    },
}

pub enum ScenarioRef {
    /// A `scenarios {}` preset in the model file.
    Named(String),
    /// An inline patch (a batch `[[scenario]]` carrying enable/disable/params).
    Inline { name: String, enable: Vec<String>, disable: Vec<String>,
             params: IndexMap<String, f64> },
}

pub enum Seeds {
    /// One base seed; replicates derive seeds by XOR-mixing.
    Single(u64),
    /// An explicit list (`--seeds`, batch `[config].seeds`), used verbatim.
    Explicit(Vec<u64>),
}

pub enum ObsOutput { None, File(PathBuf), Dir(PathBuf), OnlyFile(PathBuf), OnlyDir(PathBuf) }
```

`--seeds` accepts an inclusive range `1:100` or a comma list `1,2,42`.
`--scenario` accepts a comma-separated list or repetition; both expand to the
same `Vec<ScenarioRef>`.

Output shape is supplied by the front-end as an `engine::RunSink`: `simulate`
composes a store sink with a combined-TSV mirror sink; `batch run` uses the
store sink alone.

### 5.3 Total Runs Calculation

```
Total runs = |param_points| × |scenarios| × |seed slots|

where |param_points| =
  Point:   1
  Sweep:   product of |sweep_i.expand()| over swept parameters
  Draws:   number of draw rows

and |seed slots| =
  --seeds / [config].seeds given:  the explicit seed count
  otherwise:                       `replicates` (default 1), seeds derived from --seed

and |scenarios| =
  none specified:  1 (implicit baseline)
  otherwise:       the number of --scenario names or [[scenario]] entries
```

`--seeds` and `--replicates` are mutually exclusive; with `--seeds` the
replicate count tracks the seed-list length.

### 5.4 Scenario × Parameter-Point Interaction

Scenarios and parameter points are orthogonal axes: their Cartesian product is
the run grid, and each (point, scenario) pair is one cell.

They are **not** independent in value resolution. A scenario's `set`/`scale`
sits at a higher precedence tier than the draw/sweep point (§4.2), so:

> A parameter that a scenario `set`s takes the scenario's value, discarding the
> sweep point or draw row for that parameter.

This matters because the implicit scenario is `baseline`, and if the model
declares a preset named `baseline` that pins parameters, that preset applies
even though no `--scenario` was given. The failure is silent and total:

```bash
# sir_basic.camdl's `baseline` preset sets beta, gamma, N0, I0.
camdl simulate sir_basic.camdl --scenario baseline --draws uniform -n 3 --backend ode
```

produces three leaves with three distinct `run_id`s and three **byte-identical**
trajectories. Dropping `--scenario baseline` gives three different trajectories.
The same holds for a batch `[sweep]` whose parameters the effective scenario
pins.

Practical rule: when sweeping or drawing a parameter, the scenario in force must
not `set` it. Check with `camdl batch run --dry-run`, which prints each
scenario's resolved `set={…}` next to the sweep grid, or compare two cells'
`traj.tsv`.

### 5.5 Batch TOML Reference

The manifest is a `[config]` table plus optional `[[scenario]]`, `[sweep]` or
`[design.*]`, `[obs]`, and `[output]` sections. Unknown keys are a hard error.

```toml
[config]
model = "seir_vaccine.camdl" # .camdl source or a compiled .ir.json (required)
params = "params.toml" # optional base parameter TOML
geo = "geo/" # optional
backend = "chain_binomial" # gillespie | chain_binomial | ode
dt = 1.0
output_dir = "results"
parallel = 4 # 0 = all cores
seeds = { n = 5 } # see below
```

`seeds` accepts exactly one of:

```toml
seeds = { n = 1000 } # 1..=1000 (or start..start+n-1 with `start`)
seeds = { n = 100, start = 500 }
seeds = { from = 1, to = 100 }
seeds = { list = [1, 2, 42] }
```

with `{ }` (or an absent `seeds`) meaning the single seed `1`.

**Scenario comparison.** Each `[[scenario]]` name is resolved against the
model's `scenarios {}` presets: a bare name matching a preset routes through the
preset; a name carrying inline `enable`/`disable`/`params` is an ad-hoc
scenario; a bare name matching nothing is an error.

```toml
# batches/scenario_comparison.toml
[config]
model = "seir_vaccine.camdl"
params = "params.toml"
backend = "chain_binomial"
dt = 1.0
output_dir = "results"
parallel = 4
seeds = { n = 5 }

[[scenario]]
name = "baseline" # a model preset

[[scenario]]
name = "with_sia" # a model preset

[[scenario]]
name = "high_coverage" # ad-hoc: inline patches, name is a label
enable = ["sia_round_1"]
params = { vacc_frac = 0.95 }

# 3 scenarios × 5 seeds = 15 runs
```

```bash
camdl batch run batches/scenario_comparison.toml
```

```
   cached IR for seir_vaccine.camdl (52996338)
Done: 15/15 runs completed. Leaves under results/sims/
```

An unresolvable scenario stops the run before any cell:

```
error: scenario 'sweep_arm' is not a model preset and defines no inline enable/disable/params.
  Available model presets: 'baseline', 'high_r0'.
  Fix: use one of the listed presets, add enable/disable/params to define an ad-hoc scenario, or use
  'baseline' for the unmodified model.
```

**Cartesian sweep.** `[sweep]` maps a parameter name to a list or a generator.
Keys are sorted before expansion, so the point order is deterministic.

```toml
# batches/r0_gamma_sweep.toml
[config]
model = "sir_basic.camdl"
params = "sir_params.toml"
seeds = { n = 3 }
output_dir = "results_sweep"

[[scenario]]
name = "sweep_arm" # ad-hoc, so it does not pin beta/gamma (§5.4)
params = { I0 = 10 }

[sweep]
beta = [0.2, 0.3, 0.4, 0.5]
gamma = { linspace = { min = 0.05, max = 0.2, n = 3 } }
# 4 × 3 = 12 points × 1 scenario × 3 seeds = 36 runs
```

The four `[sweep]` value forms:

```toml
p = [0.1, 0.3, 0.5] # explicit list
p = { linspace = { min = 0.1, max = 0.9, n = 9 } } # n linearly-spaced, endpoints inclusive
p = { logspace = { min = 0.001, max = 0.1, n = 5 } } # n log-spaced, endpoints inclusive; min > 0
p = { range = { min = 1.0, max = 5.0, step = 0.5 } } # min, min+step, … ≤ max; step defaults to 1.0
```

**Space-filling design.** `[design.NAME]` replaces `[sweep]` (the two are
mutually exclusive) and generates points by `sobol`, `lhs`, or `random` over
per-parameter ranges, optionally on a `log`/`logit` scale, optionally weighted
by a `prior`.

```toml
# batches/design_lhs.toml
[config]
model = "sir_basic.camdl"
params = "sir_params.toml"
seeds = { list = [1, 2] }
output_dir = "results_design"
parallel = 2

[[scenario]]
name = "explore"
params = { N0 = 1000 }

[design.wide]
method = "lhs"
n = 6

[design.wide.parameters]
beta = { range = { min = 0.1, max = 0.6 }, transform = "log" }
gamma = { range = { min = 0.05, max = 0.3 } }

[obs]
enabled = false

[output]
no_flows = true
```

```bash
camdl batch run batches/design_lhs.toml
```

```
Design 'wide': method=lhs n=6 parameters=2
  Generated 6 parameter points
  Wrote results_design/designs/wide/parameter_points.tsv
Design 'wide' complete: 12/12 cells. Leaves under results_design/sims/.
```

The generated points are archived as
`<output_dir>/designs/<name>/parameter_points.tsv`; the simulation cells are
ordinary `sims/` leaves and dedupe against identical non-design runs.

Declaring both sections is refused:

```
error: [sweep] and [design.*] are mutually exclusive.
  [sweep] — deterministic grid for specific parameter values
  [design.*] — space-filling for sensitivity/VOI analysis
  Use one or the other in a single experiment file.
```

**Synthetic observations for the whole grid.** `[obs] enabled = true` samples
every run's observation streams into the leaf's `obs/` subtree. There is no
per-file path option — the leaves are the output. A model with no
`observations {}` block is an error rather than a silent no-op.

```toml
[config]
model = "sir_two_patch_long_obs.camdl"
seeds = { from = 1, to = 3 }
output_dir = "results_obs"

[[scenario]]
name = "baseline"

[obs]
enabled = true
```

Each leaf then carries
`obs/<obs-hash>-<obs-seed>/{cases_urban.tsv,
cases_rural.tsv, obs.json}` beside
`traj.tsv`.

**Column view.** `[output]` takes the same three keys as the `simulate` flags
and applies them to every cell:

```toml
[output]
every = 7 # like --output-every
no_flows = true # like --no-flows
columns = ["S", "I"] # like --columns
```

**CLI overrides.** `camdl batch run` accepts `--output-dir`, `--parallel`,
`--dry-run`, `--force`, and `--allow-degenerate-rates`; the first two override
the manifest's values.

### 5.6 Execution Flow

`camdl batch run`:

1. Parses the manifest (unknown keys are fatal) and resolves the model to IR,
   compiling the `.camdl` if needed.
2. Loads `[config].params`. A params file that carries an inference-provenance
   header is checked against its recorded content hash and the result reported;
   a header-less file is loaded silently.
3. Resolves the seed list, every `[[scenario]]` against the model's presets, and
   the `[output]` column view.
4. Expands `[sweep]` or `[design.*]` into parameter points and builds the
   `param-point × scenario × seed` grid.
5. Classifies each cell against the store: a committed leaf with the same
   `run_id` is skipped unless `--force`. `--dry-run` stops here and prints the
   scenario table, the point list with each value's provenance, the hit/miss
   counts, and the leaf paths.
6. Runs the remaining cells on a scoped rayon pool sized by `--parallel` (`0` =
   all cores), then merges results in canonical order — parallelism never
   perturbs a trajectory, because each cell's seed is derived from its
   coordinates, not from iteration order.
7. Writes `traj.tsv` + `run.json` (+ `obs/` when enabled) per leaf. There is no
   batch manifest file; the per-leaf `run.json` records are the index, with
   `<root>/index.json` as a derived, rebuildable lookup cache.
8. Archives `model.ir.json` at `<output_dir>/`, plus `model.render.json` and
   `model.graph.json` when the manifest pointed at `.camdl` source.

`camdl batch status <manifest>` re-resolves every cell's identity the same way
and reports how many leaves are already committed, listing the remaining cells.
It requires `[config].model` to be a compiled `.ir.json`; against a `.camdl`
source it reports `(cannot parse model …: IR JSON parse error …)` and no counts.

## 6. FitConfig — the inference type

### 6.1 Overview

A `fit.toml` specifies a single inference task: which model to fit, what data to
fit it to, which parameters to estimate versus hold fixed, and what inference
algorithm to run. It defines a _view_ of the parameter space — the partition of
the model's parameter set into free parameters (explored by the algorithm) and
fixed parameters (held constant). The algorithm then operates in the reduced
space of free parameters. (See Buffalo 2026 for the formal treatment of
parameter views, transforms, and the downward chain from inference coordinates
to simulator output.)

The runtime type is `FitConfigV2` (`rust/crates/cli/src/fit/config_v2.rs:26`).
It is the only fit-config schema, and `camdl fit run` — the sole entry point
that executes a fit — deserializes it directly.

Three properties govern the whole surface:

- **Strict parsing.** `FitConfigV2` and every nested config struct carry
  `#[serde(deny_unknown_fields)]`, so a misplaced or misspelled key is a load
  error, not a silent drop. The one exception is `[fixed]`, whose
  `#[serde(flatten)]` map of arbitrary `param = value` entries is incompatible
  with the attribute. `[stages.*]` blocks need a separate mechanism — see §6.5.
- **Paths in the file anchor at the file.** `[model] camdl`, `output_dir`,
  `[data] file`, `[data.observations]`, and `[data.holdout]` are resolved
  against the `fit.toml`'s own directory at load time
  (`config_v2.rs:2361`–`2398`), so a config is relocatable as a unit and runs
  the same from any working directory. Three path-bearing keys are **not** in
  that set and anchor at the process working directory instead:
  `[fixed]
  from_file`, `[synthetic] true_params`, and a stage's `survey_path`.
- **Absolute paths warn.** `absolute_path_warnings` (`config_v2.rs:2314`) emits
  one stderr line per absolute reference in `[model] camdl`, `output_dir`,
  `[data] file`, or `[data.observations]`, because an absolute path pins the
  config to one machine's layout. It is a warning, not an error.

### 6.2 Top-level keys

Every key `FitConfigV2` accepts at the top level. Anything else fails at load
with the serde list quoted in §6.9.

| key              | type                        | required | default                | meaning                                                                                                                                                                                                                                                                                                                                         |
| ---------------- | --------------------------- | -------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[model]`        | table                       | **yes**  | —                      | `camdl = "<path>"`. Accepts a `.camdl` source or a pre-compiled `.ir.json`.                                                                                                                                                                                                                                                                     |
| `[data]`         | table                       | one of   | —                      | Real-data source. Mutually exclusive with `[synthetic]`; exactly one must be present.                                                                                                                                                                                                                                                           |
| `[synthetic]`    | table                       | one of   | —                      | Generate N datasets from known truth and fit each (simulation-based calibration).                                                                                                                                                                                                                                                               |
| `[estimate]`     | table of tables             | **yes**  | —                      | The free parameters. See §6.3.                                                                                                                                                                                                                                                                                                                  |
| `[fixed]`        | table                       | **yes**  | —                      | The held-constant parameters. See §6.4.                                                                                                                                                                                                                                                                                                         |
| `[stages.<n>]`   | table of tables             | **yes**  | —                      | The inference pipeline, executed in declaration order. See §6.5.                                                                                                                                                                                                                                                                                |
| `[config]`       | table                       | no       | `{ dt = 1.0 }`         | Fit-wide simulator settings. See the sub-table below.                                                                                                                                                                                                                                                                                           |
| `output_dir`     | string                      | no       | `results`              | Output root. Anchored at the `fit.toml`. Not part of the fit identity.                                                                                                                                                                                                                                                                          |
| `fit_seeds`      | list of ints                | no       | `[--seed]` (CLI, or 1) | One fit per listed seed. Duplicates rejected.                                                                                                                                                                                                                                                                                                   |
| `simplex_groups` | array of tables             | no       | `[]`                   | `[[simplex_groups]] params = ["a","b",…]` — members must form a probability simplex. Honored by IF2 only; other algorithms warn.                                                                                                                                                                                                                |
| `fit_starts`     | `"model_default"`/`"prior"` | no       | `model_default`        | **Inert.** Parsed and hashed, but no runner reads it (see §6.9, note).                                                                                                                                                                                                                                                                          |
| `scenario`       | string                      | no       | none                   | Named scenario from the model; applies its enable/disable lists and param overrides before inference. Exclusive with `enable`/`disable`.                                                                                                                                                                                                        |
| `enable`         | list of strings             | no       | `[]`                   | Ad-hoc intervention enable list; `"*"` enables every toggleable intervention.                                                                                                                                                                                                                                                                   |
| `disable`        | list of strings             | no       | `[]`                   | Ad-hoc disable list. Explicit disable beats `always_active`.                                                                                                                                                                                                                                                                                    |
| `ic_free`        | bool                        | no       | `false`                | Condition the likelihood on `y₁` rather than on a committed initial state. Requires particles that differ in x₀: `if2` always, `pfilter`/plain `pmmh` only when the model's `init { }` declares a law (gh#732). Also requires a non-missing `y₁`, and — when the model's `init { }` is deterministic — a `perturb_only_at_t0 = true` parameter. |
| `condition_from` | string **or** table         | no       | none                   | Conditioning boundary for a covariate-informed burn-in. See below.                                                                                                                                                                                                                                                                              |
| `[provenance]`   | table                       | no       | none                   | Lineage metadata. See §6.7.                                                                                                                                                                                                                                                                                                                     |

`[config]` (`FitBackendConfig`, `config_v2.rs:294`):

| key                      | type                  | default         | meaning                                                                                                      |
| ------------------------ | --------------------- | --------------- | ------------------------------------------------------------------------------------------------------------ |
| `dt`                     | float                 | `1.0`           | Integrator step. This is the only honored `dt`; a top-level `dt` is a load error.                            |
| `obs_alignment`          | `"exact"` \| `"snap"` | algorithm's own | How observation times relate to the `dt` grid. Resolved per algorithm (§6.5); an unsupported request errors. |
| `allow_degenerate_rates` | bool                  | `false`         | Treat a numerical collapse in a rate expression (div-by-zero, `Pow`→NaN, `Sqrt` of a negative) as `0.0`.     |

`[config] backend` was relocated to `[synthetic] backend` (gh#241) and is
rejected with a migration message.

`[data]` (`DataSpec`, `config_v2.rs:340`):

| key             | type            | meaning                                                                                                       |
| --------------- | --------------- | ------------------------------------------------------------------------------------------------------------- |
| `file`          | string          | Single wide TSV holding one column per model-declared observation stream. Exclusive with `observations`.      |
| `observations`  | map name → path | Per-stream file paths, keyed on the `observations { }` block names in the `.camdl`. Exclusive with `file`.    |
| `holdout_after` | float           | Accepted and mutually-exclusive-checked, but **no fit-path consumer reads it**. Exclusive with `holdout`.     |
| `holdout`       | map name → path | Accepted; the files' bytes are digested into the fit `run_id`, but **no fit stage withholds or scores them**. |

Exactly one of `file` / `observations` must be set. The observation model
(likelihood family) and projection (which flow or compartment is accumulated)
are declared in the `.camdl`; `fit.toml` supplies only the file paths.

Out-of-sample evaluation today is `camdl data split` (which writes separate
train/holdout TSVs) followed by a second fit and `camdl compare`.
`[data]
holdout` / `holdout_after` do not perform it.

`[synthetic]` (`SyntheticSpec`, `config_v2.rs:453`):

| key           | type                               | required | default          | meaning                                                                           |
| ------------- | ---------------------------------- | -------- | ---------------- | --------------------------------------------------------------------------------- |
| `true_params` | string                             | **yes**  | —                | Flat TOML of `name = value` ground truth. Anchored at the working directory.      |
| `sim_seeds`   | `"N:M"` range **or** list of ints  | **yes**  | —                | One dataset per seed. Duplicates rejected; a malformed range (`"1-20"`) errors.   |
| `datasets`    | int                                | no       | `len(sim_seeds)` | Must equal `len(sim_seeds)` when supplied.                                        |
| `scenario`    | string                             | no       | none             | Scenario applied during **generation** only, not during fitting.                  |
| `backend`     | `chain_binomial`/`gillespie`/`ode` | no       | `chain_binomial` | Forward backend used to generate the datasets. Distinct from a stage's `backend`. |

`condition_from` places a leading reset-only hole on an incidence stream's
observation grid: the span `[t_start, cond_from)` is simulated with the full
dynamics but scored nowhere, the incidence accumulator resets at the boundary,
and the first scored bin is `(cond_from, first_obs]`. Two surface forms
(`ConditionFrom`, `config_v2.rs:205`):

```toml
condition_from = "first_obs - 1 week" # one default for every stream
```

```toml
[condition_from] # per-stream shadows
default = "first_obs - 1 week"
es = "first_obs - 2 weeks"
```

Each value is a bare model-time number (`"14"`), an absolute date
(`date("2020-02-01")` or a bare ISO date), or a relative offset off _that_
stream's first observation. Resolution per stream is shadow → `default` → none.
`default` is a reserved key. `condition_from` and `ic_free` cannot be combined.

`camdl pfilter`, `camdl profile`, and `camdl fit predict` all carry the same
per-stream window, so a fixed-θ loglik and a predictive row cover the window the
fit scored. `fit predict` emits no row at the boundary itself (it is a reset,
not an observation), and its free-forward projection reads the recorded
cumulative flow there — so `condition_from` must also be a recorded **output**
time, or `fit predict` refuses and names the fix.

### 6.3 `[estimate]` — free parameters

Each entry is `[estimate.<param>]` (or an inline table under `[estimate]`).
`EstimateSpecV2` (`config_v2.rs:607`):

| key                  | type                                 | default                       | meaning                                                                                                                                                                                                                                                                                                                      |
| -------------------- | ------------------------------------ | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bounds`             | `[lo, hi]`                           | the model's `in [lo, hi]`     | Search box. May only **narrow** the model's declared range; loosening is an error.                                                                                                                                                                                                                                           |
| `start`              | float                                | model value, else bounds draw | The stage's base point. An upstream stage's result (`init_mle`) overrides it.                                                                                                                                                                                                                                                |
| `prior`              | inline table                         | the model's `~` declaration   | See the wire format below. Required in some form for `pgas`/`pmmh`/`mh`/`nuts`.                                                                                                                                                                                                                                              |
| `transform`          | `"log"` \| `"logit"` \| `"identity"` | derived from the param's type | Inference-scale transform. Also sets the clamp box IF2 keeps particles inside.                                                                                                                                                                                                                                               |
| `perturb_only_at_t0` | bool                                 | `false`                       | Perturb at t=0 only, not at every observation — the IF2 schedule for an initial-state parameter. Read by `if2` stages and ignored by the rest; a config-load error only when the fit declares no `if2` stage at all. Required by `ic_free` unless the model's `init { }` declares a law, which supplies the same t=0 spread. |
| `rw_sd`              | float                                | auto-scaled from bounds       | IF2 per-parameter random-walk SD, on the natural scale.                                                                                                                                                                                                                                                                      |

An entry with no fields at all (`beta = {}`) is legal: it means "estimate this,
take everything from the model."

**Transform defaults** (`runner.rs:1007`, `derive_transform_with_bounds`), keyed
on the parameter's declared type in the `.camdl`:

| declared type                           | derived transform                                    |
| --------------------------------------- | ---------------------------------------------------- |
| `probability`                           | `logit`, clamped to the resolved bounds              |
| `rate`, `positive`, `count`, `duration` | `log`, clamped to the resolved bounds                |
| `instant`, `real`                       | `logit` when both bounds are finite, else `identity` |
| (no declared type)                      | `log` when `lo >= 0`, else `identity`                |

The clamp uses the _resolved_ bounds, so narrowing in `fit.toml` genuinely
narrows the IF2 search rather than leaving it advisory.

**Prior wire format.** Externally tagged inline tables, matching what the OCaml
compiler emits for in-model `~` priors (`PriorDist`,
`rust/crates/ir/src/parameter.rs:32`). Field names are exact — `normal` takes
`mean`/`sd`, `log_normal` takes `mu`/`sigma`:

```toml
prior = { uniform = { lower = 0.0, upper = 1.0 } }
prior = { uniform = {} } # over the param's bounds
prior = { normal = { mean = 0.0, sd = 1.0 } }
prior = { log_normal = { mu = -1.2, sigma = 0.5 } }
prior = { half_normal = { sigma = 1.0 } }
prior = { beta = { alpha = 2.0, beta = 5.0 } }
prior = { gamma = { shape = 2.0, rate = 1.0 } }
prior = { exponential = { rate = 1.0 } }
prior = { log_uniform = { lower = 1e-5, upper = 1e-2 } }
prior = { truncated_normal = { mean = 0.7, sd = 0.2, lower = 0.3, upper = 1.0 } }
prior = { flat = {} } # fit.toml only
```

`{ uniform = {} }` is uniform over the parameter's resolved bounds (fit `bounds`
falling back to the model's `in [lo, hi]`); it errors when neither supplies
them. `truncated_normal`'s `lower`/`upper` must equal the resolved bounds — the
prior's support and the search box are one interval. `{ flat = {} }` has no DSL
counterpart: it is the accountable opt-in to an improper-uniform prior, recorded
in provenance as `flat_explicit`.

**Prior precedence** (three tiers, `validate_priors_present`,
`config_v2.rs:2809`): a `fit.toml` `prior` wins over the model's `~`
declaration; an explicit `{ flat = {} }` counts as declared. A Bayesian stage
whose estimated parameter has none of the three is a **hard error** — `fit run`
never falls back to flat silently, because downstream consumers treat the chain
as the canonical posterior. (`camdl profile` warns instead of erroring; the bar
there is lower.)

**Prior × transform compatibility** (`runner.rs:2813`) is enforced before any
fitting. `log_normal`, `half_normal`, `gamma`, `exponential`, and `log_uniform`
require a `log` transform; `beta` requires `logit` with bounds exactly `[0, 1]`;
`flat`, `uniform`, `normal`, and `truncated_normal` accept any transform.

### 6.4 `[fixed]` — held-constant parameters

`FixedParams` (`config_v2.rs:770`). Three sources, and the union of `[estimate]`
and `[fixed]` must be exactly the model's parameter set:

| key             | type   | meaning                                                                                            |
| --------------- | ------ | -------------------------------------------------------------------------------------------------- |
| `from_file`     | string | Bulk-load a flat TOML of `name = value`. Resolved against the **working directory**, not the file. |
| `from_scenario` | string | Bulk-load a named `scenario` block from the `.camdl`, following its `compose = [...]` chain.       |
| `<param> = <n>` | float  | Inline values. Override `from_file` entries on collision.                                          |

`from_scenario` is mutually exclusive with both `from_file` and inline values:
allowing an override would make the fit a hybrid that corresponds to nothing in
the `.camdl`. The one carve-out is structural — a scenario parameter that also
appears in `[estimate]` is skipped on import, so a single `baseline` scenario
can serve both forward simulation and a fit's `[fixed]`.

### 6.5 `[stages.<name>]` — the inference pipeline

Stages are user-named and execute in declaration order. Each block is tagged by
`algorithm` (`Stage`, `config_v2.rs:987`, `#[serde(tag = "algorithm")]`) and
carries an explicit `backend`. Each algorithm is valid on exactly one backend;
the pair is checked at load against the `METHODS` registry
(`rust/crates/cli/src/fit/methods.rs:68`), which is the single source of truth
for `camdl fit methods`, the runtime banners, and the invalid-pair error.

| `algorithm` | `backend`        | status | category   | role                                                             |
| ----------- | ---------------- | ------ | ---------- | ---------------------------------------------------------------- |
| `if2`       | `chain_binomial` | stable | inference  | Iterated filtering → MLE.                                        |
| `pgas`      | `chain_binomial` | stable | inference  | Particle Gibbs + NUTS on θ → posterior. Default Bayesian path.   |
| `pmmh`      | `chain_binomial` | stable | inference  | Pseudo-marginal MH → posterior. Degrades for T > 500.            |
| `pfilter`   | `chain_binomial` | stable | diagnostic | Bootstrap particle filter at fixed θ → loglik, ESS, prequential. |
| `nl-sbplx`  | `ode`            | beta   | inference  | NLopt Sbplx → deterministic MLE. Robust at bounds.               |
| `nl-bobyqa` | `ode`            | beta   | inference  | NLopt BOBYQA → deterministic MLE. Faster, fails at bounds.       |
| `mh`        | `ode`            | beta   | inference  | Gradient-free MH on the ODE marginal likelihood → posterior.     |
| `nuts`      | `ode`            | beta   | inference  | NUTS via forward sensitivities on the ODE marginal → posterior.  |

The `ode`-backend samplers fit the deterministic marginal likelihood
`p(y | θ, ODE skeleton)`, not the stochastic `p(y | θ)` — a different
statistical object, appropriate for equilibrium or large-population models.
`gillespie` is a forward-simulation backend with no inference interface and is
rejected at parse.

Keys common to every stage: `algorithm`, `backend`, `init_mle` (§6.6). All
algorithms except `pfilter` and the two NLopt variants also take `init`,
`survey_path`, and `survey_top_k_n`; the NLopt variants take them in the
`fit.toml` but ignore the corresponding CLI overrides.

**`if2`** — `chains`, `particles`, `iterations`, `cooling` required.

| key                    | default | meaning                                                                                                                                                                                                              |
| ---------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cooling`              | —       | Fraction of the initial perturbation SD remaining at `cooling_target_iters`.                                                                                                                                         |
| `cooling_target_iters` | `50`    | pomp's `cooling.fraction.50` convention; decoupled from `iterations`.                                                                                                                                                |
| `loglik_eval`          | table   | Clean re-scoring of candidate θ̂. `n_particles = 4000`, `n_replicates = 8`, `combine = "log_mean_exp"` (or `"mean"`).                                                                                                 |
| `gate`                 | table   | `a_thresh = 1.01` (chain-agreement Â ceiling), `decibans_thresh = 30.0` (inter-chain loglik spread floor).                                                                                                           |
| `dt_check`             | table   | Post-fit Richardson dt-convergence audit. `enabled = true`, `n_halvings = 2`; `n_particles`/`n_replicates`/`combine` inherit `loglik_eval`; `threshold_nats` defaults per backend (2.0 chain_binomial, 0.5 ode_rk4). |

`iterations` must be ≥ 1. IF2 has no extension dimension, so `--resume` does not
apply: its cooling schedule is determined by the total iteration count, and
restarting mid-schedule is statistically incoherent.

**`pgas`** — `chains`, `particles`, `sweeps` required.

| key                    | default | meaning                                                                                        |
| ---------------------- | ------- | ---------------------------------------------------------------------------------------------- |
| `burn_in`              | `2000`  | Discarded sweeps. Must be `< sweeps`.                                                          |
| `thin`                 | `5`     | Retain every k-th post-burn-in sweep.                                                          |
| `tempering`            | `[1.0]` | Parallel-tempering ladder of β ∈ (0,1]. First entry must be `1.0`; only the cold rung samples. |
| `max_tree_depth`       | `10`    | NUTS tree-depth ceiling for the θ\|X update.                                                   |
| `trajectory_warmup`    | `0`     | CSMC-only sweeps before parameter updates begin.                                               |
| `csmc_sweeps_per_nuts` | `1`     | CSMC trajectory updates per parameter update.                                                  |
| `n_trajectories`       | `200`   | Posterior trajectories written to disk. Output-shaping, but keyed.                             |
| `dense_mass`           | `true`  | Full-covariance NUTS metric; `false` for diagonal.                                             |
| `use_nuts`             | `true`  | `false` falls back to MH-within-Gibbs for θ\|X.                                                |

**`pmmh`** — `chains`, `particles`, `iterations` required.

| key           | default | meaning                                                                              |
| ------------- | ------- | ------------------------------------------------------------------------------------ |
| `burn_in`     | `5000`  | Must be `< iterations`.                                                              |
| `thin`        | `10`    |                                                                                      |
| `adapt`       | `true`  | Haario adaptive Metropolis on the proposal SDs.                                      |
| `adapt_start` | `300`   | MCMC step at which adaptation begins.                                                |
| `rho`         | none    | Crank–Nicolson correlation for correlated pseudo-marginal MCMC; must be in `[0, 1)`. |

**`mh`** — `chains`, `iterations` required. Reuses the PMMH chain and
adaptive-proposal machinery with the particle-filter likelihood swapped for the
deterministic ODE evaluation, so it carries neither `particles` nor `rho`.

| key           | default | meaning                                                                                                                             |
| ------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `burn_in`     | `5000`  | Must be `< iterations`.                                                                                                             |
| `thin`        | `10`    |                                                                                                                                     |
| `adapt`       | `true`  |                                                                                                                                     |
| `adapt_start` | `300`   |                                                                                                                                     |
| `burnin_dt`   | none    | Coarse RK4 step for the unscored warm-up `[t_start, first_obs)`. Must be `> dt`; prevalence-scored streams only. Identity-defining. |

**`nuts`** — `chains` required.

| key              | default | meaning                                                                   |
| ---------------- | ------- | ------------------------------------------------------------------------- |
| `warmup`         | `500`   | Step-size adaptation draws, discarded.                                    |
| `samples`        | `500`   | Posterior draws kept per chain.                                           |
| `max_tree_depth` | `10`    |                                                                           |
| `target_accept`  | `0.8`   | Dual-averaging target (Stan's default).                                   |
| `dense_mass`     | `false` | Diagonal metric by default (Stan's `diag_e`); `true` for full covariance. |
| `burnin_dt`      | none    | As for `mh`; the state and its sensitivities are coarsened together.      |

`nuts` has no `burn_in`/`thin` — `warmup` is the discard.

**`pfilter`** — `particles` required. No `chains` (it is always one), no `init`.

| key                  | default | meaning                                                                     |
| -------------------- | ------- | --------------------------------------------------------------------------- |
| `replicates`         | none    | Independent filter passes; the stage reports loglik mean ± SD across them.  |
| `record_ancestry`    | `false` | Record per-step ancestor indices for smoothing-path reconstruction.         |
| `record_prequential` | `true`  | Record per-step predictive samples and log-likelihoods for `camdl compare`. |

**`nl-sbplx` / `nl-bobyqa`** — `backend`, `chains` required (`NloptStageConfig`,
`config_v2.rs:1340`).

| key         | default | meaning                                                              |
| ----------- | ------- | -------------------------------------------------------------------- |
| `tolerance` | `1e-6`  | NLopt `xtol_rel`.                                                    |
| `max_evals` | `5000`  | Per-chain objective-evaluation budget; hitting it is a soft failure. |
| `gate`      | table   | Two-leg convergence gate, same shape as IF2's.                       |

`chains` here is the number of independent multi-start optimizations;
`init =
"single"` defeats multi-start, since every chain then converges from the
same point.

**Observation alignment.** `[config] obs_alignment` is resolved per algorithm
(`methods.rs:495`). `if2` and `pfilter` step exactly to observation times and
reject `"snap"`. `pgas` uses a uniform grid and rejects `"exact"`. Plain `pmmh`
is exact; correlated `pmmh` (`rho` set) pre-draws one block of random numbers
per observation window, each sized at that window's own substeps, so an
irregular grid is fine — a daily reporting series with a day of no reporting
included — but the observation times must walk forward from `t_start`. The ODE
algorithms score on the integrator grid and reject the key entirely.

**IC-free support.** `ic_free = true` is honored by `if2`, and by `pfilter` /
plain `pmmh` on a model whose `init { }` declares a law
(`methods.rs::validate_ic_free`). `pgas`, correlated `pmmh` and the ODE
algorithms score every observation unconditionally, so the combination is
rejected rather than allowed to compute the unconditional likelihood under a
banner claiming otherwise. `pfilter` / plain `pmmh` on a _deterministic_
`init { }` are rejected for the other reason: their bootstrap filter draws x₀
per particle, but every such draw returns the same state, so the first reweight
has nothing to discriminate between (gh#732).

### 6.6 Chain starts: `init` and `init_mle`

Two orthogonal keys answer different questions, and both live on the stage.

**`init_mle` — where the stage's base point comes from.** Deserialized into
`StartsFrom` (`config_v2.rs:2069`) by inspecting the string: containing `/` or
`\` → an external results directory; the literal `"random"` → no upstream; any
other bare word → the name of an earlier stage in this file. Default `"random"`.
A stage reference must name a stage declared **before** this one; the DAG is
validated at load. When an upstream stage is named, its identity and consumed
`fit_state.toml` are folded into this stage's content hash, so re-running the
upstream re-keys the downstream.

**`init` — how the per-chain starts are spread around that base point.**
`InitMethod` (`rust/crates/cli/src/fit/init.rs:69`); the default is
`uniform_unconstrained` (`init.rs:167`).

| `init`                  | where the chains start                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uniform_unconstrained` | (default) i.i.d. `U(-2, 2)` on the unconstrained scale, squashed and mapped into bounds. Boundary-avoiding and scale-invariant; Stan's default. Base point unused. |
| `lhs`                   | Latin-hypercube stratified over bounds, log-scale-aware for `Log` parameters. Best full-bounds coverage at low chain counts. Base point unused.                    |
| `uniform`               | Per-chain uniform draw within natural-scale bounds. Legacy; clumps for `Log` parameters.                                                                           |
| `single`                | Every chain at the base point.                                                                                                                                     |
| `survey_top_k`          | Top-K rows of a `camdl survey` landscape. Requires `survey_path`; `survey_top_k_n` defaults to `chains` and must equal it.                                         |
| `from_prior`            | One draw per chain from each parameter's `~` declaration in the model IR.                                                                                          |
| `from_posterior`        | One row per chain, drawn uniformly with replacement from a posterior draws TSV or a fit directory's `draws.tsv`.                                                   |
| `from_mle`              | Every chain at the MLE point of a prior fit (`mle.toml`, else `final_params.toml`).                                                                                |
| `from_params`           | Every chain at the point in a hand-written flat params TOML.                                                                                                       |

`uniform`, `lhs`, and `uniform_unconstrained` collapse to the base point at
`chains = 1`; the source-reading modes do not. `from_posterior`, `from_mle`, and
`from_params` are constructed by the CLI (`--init` plus `--posterior` / `--mle`
/ `--params`) and cannot be written as bare strings in `fit.toml`.

`init` and its companions are identity-bearing: `identity_payload`
(`config_v2.rs:1526`) folds `init_method`, `survey_path`, and `survey_top_k_n`
into the stage hash, and the CLI overrides are written into the stage _before_
the content address is taken (`apply_cli_overrides`, `config_v2.rs:1722`), so
two runs differing only in `--init` are two artifacts.

Each stage writes `chain_starts.tsv` recording where every chain actually began,
before any perturbation. That file, not the config, is the authority on what a
run did.

**Base-point precedence** (`runner.rs:235`), lowest to highest: the model's
declared value, `[fixed]`, `[estimate].start`, then the upstream stage's result
via `init_mle`. The upstream result wins last so `init_mle = "scout"` is not
silently overwritten by a stale `start` left in the file.

### 6.7 `[provenance]` — lineage metadata

```toml
[provenance]
derived_from = "fits/01_all_free.toml"
reason = "free the reporting fraction"
```

Both fields are optional strings (`FitProvenance`, `config_v2.rs:2119`). No
runner reads them; `camdl fit new` writes `derived_from` when deriving one
config from another, and the block is otherwise for human navigation.

It is nonetheless **identity-bearing**: the fit-level hash covers the whole
serialized config with only `stages`, `fit_seeds`, and `output_dir` removed
(`cas.rs:304`), so editing `reason` changes the fit hash and forces a re-fit
into a new segment.

### 6.8 Where output goes

Fits are content-addressed. There are no configured output paths beyond the root
— the directory names are hashes of the inputs, and the layout is factored into
three levels (`cas.rs:1`–`24`):

```
<root>/fits/<stem>-<h8>/                          fit level
        model.ir.json  model.render.json  model.graph.json
        model.camdl.original  fit.toml.original  fit.meta.json
    <NN>-<stage>-<h8>/                            stage level
        <seed_N>-<h8>/                            seed level (the leaf)
            run.json  fit_state.toml  chain_starts.tsv
            chain_1/ … chain_N/
```

- **`<root>`** is `output_dir` from the `fit.toml` (anchored at the `fit.toml`),
  else `$CAMDL_OUTPUT_DIR` (anchored at the working directory), else `results`
  (`rust/crates/cli/src/run_paths.rs:48`). `fit run` has no `--output-dir` flag.
- **`<stem>`** is the `fit.toml`'s file stem; **`<h8>`** at the fit level is the
  first eight hex digits of the fit hash — the whole-IR model digest, the
  per-stream training and holdout data digests, the canonicalized config (less
  `stages` / `fit_seeds` / `output_dir`), and the engine version.
- **`<NN>`** is the stage's zero-padded topological position; the stage `<h8>`
  folds the stage's `identity_payload` with its `deps`, so `02-posterior` keys
  on `01-scout`'s identity.
- **The leaf** is per seed. `run.json` is the CAS record; there is no manifest
  and no fit-wide `run.json` — the fit level is a path segment, and
  `fit.meta.json` is its sidecar (label, model hash, config archive).

Per-algorithm leaf contents: IF2 writes `mle_params.toml`, `final_params.toml`,
`diagnostics.tsv`, `chain_evaluations.tsv`, and
`chain_N/{final_params.toml, parameter_traces.tsv}`. The samplers write
`draws.tsv`, `<algorithm>_summary.json` (`pgas_summary.json`,
`pmmh_summary.json`, `mh_summary.json`, `nuts_summary.json`), and
`chain_N/trace.tsv`; PGAS and PMMH also write `chain_N/resume_state.bin`. The
NLopt stages write `mle_params.toml` and `chain_results.tsv`. A `pfilter` stage
with `record_prequential` writes `prequential.{tsv,json}`.

A completed leaf is reused on a second identical invocation ("cache hit"); pass
`--force` to re-run and overwrite.

Fits are addressed downstream by handle (`handle.rs:26`): `@label`, a `fit.toml`
path, a run directory, or a fit-hash prefix. Ambiguity is listed git-style
rather than guessed.

### 6.9 Load-time validation

`FitConfigV2::from_toml_str` (`config_v2.rs:2291`) runs three passes before the
typed parse or on its result; `FitConfigV2::validate` (`config_v2.rs:2548`) then
runs the semantic checks once the model IR is loaded and `[fixed]
from_scenario`
has been expanded (`mod.rs:312`–`327`). Everything below fires before any stage
executes.

**Parse-time.**

- A relocated `[config] backend`:

  > `` `[config].backend` has moved to `[synthetic].backend` (gh#241). ``

- The renamed stage keys, bundled so all offenders are fixable in one pass:

  > `fit.toml uses legacy stage keys removed in CLI UX rev 2 …`
  > `error: legacy key`init_method`on stage(s):`s``> `  replacement: rename to `init` (matches CLI `--init`).`
  > `  error: legacy key `starts_from` on stage(s): `s``
  > `replacement: rename to`init_mle`(one toml key per concept).`

- An unknown top-level key:

  > `` unknown field `dt`, expected one of `model`, `data`, `synthetic`, `fit_seeds`, `simplex_groups`, `fit_starts`, `output_dir`, `estimate`, `fixed`, `stages`, `config`, `scenario`, `enable`, `disable`, `ic_free`, `condition_from`, `provenance` ``

- An unknown key inside a `[stages.*]` block. `Stage` is internally tagged, so
  serde cannot deny unknown fields on it; a post-parse pass compares the raw
  keys against the set the parsed stage serializes back to
  (`validate_stage_keys`, `config_v2.rs:2251`):

  > ``unknown key `chains` in [stages.s] (algorithm = "pfilter").``
  > `allowed keys: algorithm, backend, init_mle, particles, record_ancestry, record_prequential, replicates`

  The two NLopt variants are newtype-wrapped structs and get plain serde
  `deny_unknown_fields` instead:
  `` unknown field `zzz`, expected one of `backend`, `chains`, `tolerance`, `max_evals`, `init_mle`, `init`, `survey_path`, `survey_top_k_n`, `gate` ``.

- Unknown enum values — an unrecognized `algorithm` (via a missing tag),
  `backend`
  (`` unknown variant `gillespie`, expected `chain_binomial` or
  `ode` ``),
  `init`, or a prior whose field names do not match a catalogue entry
  (`data did not match any variant of untagged enum EstimatePriorSpec`).

**Semantic, in the order `validate` applies them.**

1. Exactly one of `[data]` / `[synthetic]`.
   > `fit config has neither [data] nor [synthetic] — one must be supplied.`
2. `[synthetic]` internal consistency — non-empty `sim_seeds`, no duplicates,
   `datasets == len(sim_seeds)`.
3. `[data]` exactly one of `file` / `observations`.
   > ``[data]: `file = "..."` and `[data.observations]` are mutually exclusive — choose one.``
4. `fit_seeds` non-empty and duplicate-free.
   > `duplicate fit_seed 1 — each seed must be unique to avoid provenance-hash collisions between fits`
5. `scenario` exclusive with `enable`/`disable`.
6. `holdout_after` exclusive with `holdout`.
7. **Partition.** `[estimate] ∩ [fixed] = ∅`, `[estimate] ∪ [fixed]` equals the
   model's parameter set exactly. Three distinct errors:
   > `parameters in both [estimate] and [fixed]: beta`
   > `parameters neither estimated nor fixed: I0`
   > `parameters not in model: nope`
8. **(algorithm, backend)** against the registry. The error names the structural
   reason, suggests the right alternative, and lists the supported pairs:
   > `stage 's': stage has algorithm = "if2" with backend = "ode", which is not a supported inference method.`
9. `ic_free` support per stage, and `ic_free` exclusive with `condition_from`.
10. IF2 `iterations ≥ 1`.
11. `burn_in < iterations` (`pmmh`, `mh`) and `burn_in < sweeps` (`pgas`), using
    the **defaults** when unset — so a short sampler run must set `burn_in`
    explicitly:
    > `stage 's': burn_in (2000) ≥ sweeps (5) — every sample is discarded as burn-in, so the fit retains no posterior draws (and the post-burn acceptance rate degenerates to 0%). Reduce burn_in or raise sweeps. (burn_in defaults to 2000 when unset.)`
12. **Stage DAG.** `init_mle` must name a declared, earlier stage. Both errors
    print `starts_from` — the internal Rust field name
    (`fit/config_v2.rs:2956`), not the TOML key. The key to edit is `init_mle`;
    writing `starts_from` is rejected as a legacy key (§7.2).
    > `stage 'a': starts_from = "nope" does not match any stage.`
    > `stage 'a': starts_from = "b" but 'b' is declared after 'a'.`
13. Non-empty `bounds` on every entry that declares them.
    > `estimate.beta: bounds [0.6, 0.15] are empty (lo must be < hi)`
14. Simplex groups: ≥ 2 members, every member in `[estimate]`, no member in two
    groups, no member with `perturb_only_at_t0 = true`, no negative lower bound.

`validate_priors_present` runs next, with the model IR in scope so the `~`
fallback is honored; then two warnings that do not stop the run — priors
declared but consumed by no stage, and a posterior sampler using
`init =
"single"` with `chains > 1` (which makes R̂ uninformative).

Two checks fire later, at stage build rather than at load: the
bounds-may-narrow-not-loosen rule (`runner.rs:1113`) and `burnin_dt` validity
(`config_v2.rs:1894`).

> **Note — `fit_starts` is inert.** The key parses, is hashed into the fit
> identity, and suppresses the dangling-priors warning, but no runner reads it:
> setting `fit_starts = "prior"` does **not** draw chain starts from the priors.
> The working knob is `[stages.<name>] init = "from_prior"`.

## 7. The Fit CLI

`camdl fit` is a group of seven subcommands. One runs inference; three read what
a run left behind; two work on configs; one is a static capability listing.

| subcommand    | what it does                                                        |
| ------------- | ------------------------------------------------------------------- |
| `fit run`     | execute the stages declared in a `fit.toml`                         |
| `fit summary` | render one fit's convergence verdict, θ̂ table and provenance checks |
| `fit table`   | walk a results tree and render one row per fit                      |
| `fit diff`    | compare two `fit.toml` configs                                      |
| `fit new`     | scaffold a new `fit.toml` derived from an existing one              |
| `fit predict` | write the posterior-predictive (predicted-vs-observed) artifact     |
| `fit methods` | list the supported (algorithm, backend) pairs                       |

There is no `camdl fit list` and no `camdl fit where`. Browsing every cached run
— fits, simulations, profiles — is the top-level `camdl list`; attaching or
changing a display label after the fact is the top-level `camdl label`.

### 7.1 Invocations

Every command below was run against the worked project of §8 (`models/`,
`data/`, `fits/`, `results/`).

```bash
# Run every stage in the config.
camdl fit run fits/01_mle.toml --seed 1

# Long fits: force plain progress lines and capture them.
camdl fit run fits/01_mle.toml --seed 1 --progress plain 2>&1 | tee fit.log

# Run one stage by name. See the caveat below on staged pipelines.
camdl fit run fits/01_mle.toml --stage mle

# Sweep a [fixed] parameter; repeat the flag for a Cartesian grid.
camdl fit run fits/01_mle.toml --sweep "rho=0.5,0.6"
camdl fit run fits/01_mle.toml --sweep "rho=0.5,0.6" --sweep "k=5,10"
camdl fit run fits/01_mle.toml --sweep "k=lin(5,15,3)"
camdl fit run fits/01_mle.toml --sweep "k=log10(1,100,3)"

# Tag the run so it is findable later.
camdl fit run fits/01_mle.toml --label "auto rw_sd, take 1"

# Warm-start a stage from another fit's MLE leaf.
camdl fit run fits/09_pgas_only.toml --stage posterior \
    --init from_mle --mle results/fits/01_mle-2030ba2b/01-mle-fee126b1/seed_1-06cbd6b3

# Read side.
camdl fit summary results/fits/01_mle-2030ba2b
camdl fit summary '@auto rw_sd, take 1'
camdl fit summary results/fits/01_mle-2030ba2b --params-only
camdl fit table results/fits
camdl fit table results/fits --with-method pgas --format md
camdl fit diff fits/01_mle.toml fits/02_posterior.toml
camdl fit new --from fits/01_mle.toml fits/04_derived.toml
camdl fit predict fits/02_posterior.toml --n-draws 20
camdl fit methods
camdl list
```

**Fit handles.** `fit summary` and `fit predict` take a _handle_, not only a
path: `@<label>`, a fit-level hash prefix, a run directory, or the `fit.toml`
itself. A handle that maps to more than one fit is a hard error that lists the
candidates:

```
$ camdl fit summary fits/01_mle.toml --params-only
error: fits/01_mle.toml resolves to 2 fits:
    fits/../results/fits/01_mle-2030ba2b
    fits/../results/fits/01_mle-cc381195
  Pass a run directory or a longer hash prefix to disambiguate.
```

**`--stage` and staged pipelines.** `--stage <name>` reduces the run to that one
stage. A stage that declares `init_mle = "<upstream-stage>"` cannot be run this
way, even when the upstream stage's results are already on disk — the upstream
identity is only known for stages executed in the same invocation:

```
$ camdl fit run fits/02_posterior.toml --stage posterior
── stage: posterior (method=pgas) ──
error: stage 'posterior' starts_from 'scout', which has not run in this pipeline
```

Re-running the whole pipeline is cheap in that situation: completed stages are
served from the content-addressed store.

```
── stage: scout (method=if2) ──
  cache hit — reusing fits/02_posterior-77595169/01-scout-fee126b1/seed_1-06cbd6b3
```

**Two flag spellings are parsed only to reject them**, so the error can name the
replacement rather than printing `unexpected argument`:

```
$ camdl fit run fits/01_mle.toml --stage mle --starts-from results/x
error: --starts-from is no longer accepted on `camdl fit run`. Replacement:
  --init from_mle --mle <fit-dir>       (warm-start every chain from a prior fit's MLE)
  --init from_params --params <toml>    (warm-start from a hand-written params TOML)
Saw --starts-from results/x.
See `camdl fit run --help` (INIT MODES section).

$ camdl fit run fits/01_mle.toml --stage mle --init-method lhs
error: --init-method is no longer accepted on `camdl fit run`. It was renamed to --init for parity with `camdl profile`.
Saw --init-method lhs.
See `camdl fit run --help` (INIT MODES section).
```

The same rename reached the `fit.toml`. The stage key `starts_from` is rejected
with its replacement spelled out:

```
$ camdl fit run fits/bad_startsfrom.toml
error: error in fits/bad_startsfrom.toml:
fit.toml uses legacy stage keys removed in CLI UX rev 2 (proposal 2026-05-25-cli-init-and-params-ux §"fit.toml schema").

  error: legacy key `starts_from` on stage(s): `scout`
  replacement: rename to `init_mle` (one toml key per concept).
  example: `[stages.scout]\n  init_mle = "<prior-stage>"` (was: `starts_from = "<prior-stage>"`).
```

### 7.2 CLI Type

`fit run` takes the config path plus five groups of flags: run control, chain
starts, the conditioning window, the dt-convergence audit, and per-algorithm
overrides. The run-control core:

```rust
pub struct FitRunArgs {
    /// Fit configuration file (v2 TOML)
    pub config: PathBuf,

    /// Run only this stage by name
    #[arg(long)]
    pub stage: Option<String>,

    /// Rayon thread cap; 0 = all logical cores. Bit-identical regardless.
    #[arg(long, default_value_t = 0, env = "CAMDL_PARALLEL")]
    pub parallel: usize,

    /// RNG seed (default: 1)
    #[arg(long)]
    pub seed: Option<u64>,

    #[arg(long)]
    pub force: bool,

    /// Extend a completed PGAS/PMMH stage from a base run, addressed by
    /// run_id prefix or leaf path.
    #[arg(long, value_name = "BASE_REF", requires = "stage", conflicts_with = "force")]
    pub resume: Option<String>,

    /// Cartesian sweep over a [fixed] parameter (repeatable).
    /// SPEC is `V1,V2,...` | `lin(min,max,n)` | `log10(min,max,n)`.
    #[arg(long, value_name = "NAME=SPEC")]
    pub sweep: Vec<SweepSpec>,

    /// Proceed even if the prior stage failed its convergence gate.
    #[arg(long)]
    pub allow_nonconverged_scout: bool,

    /// Display label, 1–64 chars of `[A-Za-z0-9 ,._-]`.
    #[arg(long, value_name = "TEXT")]
    pub label: Option<String>,
}
```

Note the shapes. `--resume` carries a **base reference**, not a boolean;
`--parallel` is a plain `usize` with a `0` default meaning "all cores"; there is
**no `--output-dir`** (the output root is the `fit.toml`'s `output_dir` key, or
`CAMDL_OUTPUT_DIR`, or `./results`) and **no `--skip-chains`** (chain exclusion
is a read-side view: `fit summary --exclude-chains`,
`fit predict --exclude-chains`, `fit table --exclude-chains`).

> **`--force` does not currently work on `fit run`.** On a fit whose stage leaf
> already exists it aborts rather than recomputing:
>
> ```
> $ camdl fit run fits/08_holdoutfiles.toml --seed 1 --force
> ── stage: scout (method=if2) ──
> error: claim fit stage …/01-scout-c3b8cbc0/seed_1-06cbd6b3: artifact already completed at …/01-scout-c3b8cbc0/seed_1-06cbd6b3
> ```
>
> Without the flag the same command reports a cache hit and exits 0. To force a
> recompute today, delete the stage leaf (or the whole fit segment) first. This
> is the opposite of `camdl simulate --force`, which re-stores in place.

`--resume <BASE_REF>` extends a completed PGAS or PMMH stage: the base leaf is
read read-only and the longer chain is written to a new content-addressed leaf
keyed on the new `sweeps` / `iterations` with a dependency on the base — a
distinct deterministic artifact, not bit-identical to an uninterrupted fit of
the same length. It requires `--stage`, so it inherits the staged-pipeline
restriction described in §7.1: a PGAS stage declaring `init_mle = "<upstream>"`
cannot be resumed today. Give the resumable stage its own `fit.toml` and take
its start from a path instead of from a sibling stage:

```
$ camdl fit run fits/09_pgas_only.toml --stage posterior \
    --init from_mle --mle results/fits/01_mle-2030ba2b/01-mle-fee126b1/seed_1-06cbd6b3
   stored posterior · …/09_pgas_only-649b2ecc/01-posterior-7107153d/seed_1-06cbd6b3 · 1.5s

# raise `sweeps` in the toml, then extend that leaf
$ camdl fit run fits/09_pgas_only.toml --stage posterior \
    --init from_mle --mle results/fits/01_mle-2030ba2b/01-mle-fee126b1/seed_1-06cbd6b3 \
    --resume results/fits/09_pgas_only-649b2ecc/01-posterior-7107153d/seed_1-06cbd6b3
  chain 1: resuming from sweep 100
  chain 2: resuming from sweep 100
   stored posterior · …/09_pgas_only-649b2ecc/01-posterior-05ea6411/seed_1-06cbd6b3 · 2.3s
```

**Chain starts.** `--init <MODE>` overrides the stage's `init` and requires
`--stage`. Modes and their companion flags:

| `--init`                | companion         | where the chains start                                             |
| ----------------------- | ----------------- | ------------------------------------------------------------------ |
| `uniform_unconstrained` | —                 | default; i.i.d. U(−2, 2) on the unconstrained scale, mapped inward |
| `single`                | —                 | every chain at the seeded base point                               |
| `uniform`               | —                 | per-chain uniform draw inside `[estimate]` bounds                  |
| `lhs`                   | —                 | Latin-hypercube stratified inside bounds                           |
| `from_prior`            | —                 | one draw per chain from each parameter's `~ <dist>` declaration    |
| `from_posterior`        | `--posterior <P>` | rows from a draws TSV or a fit-results directory                   |
| `from_mle`              | `--mle <P>`       | every chain at a prior fit's MLE                                   |
| `from_params`           | `--params <TOML>` | every chain at a point in a flat params TOML                       |
| `survey_top_k`          | `--survey-path`   | top-K rows of a `camdl survey` landscape (`--survey-top-k <N>`)    |

Init applies only to parameters in `[estimate]`; anything in `[fixed]` takes its
declared value regardless of mode.

**Conditioning and the dt audit.** `--condition-from <WHEN>` mirrors the
top-level `condition_from` key and overrides it, setting the all-streams default
(per-stream shadows stay TOML-only). It accepts a model-time number (`14`), a
calendar date (`2020-02-01`), or a relative offset (`"first_obs - 1 week"`), and
a set value re-keys the fit. `--no-dt-check` skips the post-fit Richardson
dt-convergence check, `--dt-check-strict` tightens its threshold (0.5 nats for
chain_binomial, 0.1 for ode_rk4, against routine defaults of 2.0 / 0.5), and
`--dt-check-halvings <N>` sets how many halvings it evaluates (default 2).

**Per-algorithm overrides.** Each requires `--stage`. Each writes through to the
stage identity, so an overridden run is a distinct artifact rather than a cache
hit on the un-overridden one — with the exception noted after the table.

| flag                      | overrides                             | algorithm |
| ------------------------- | ------------------------------------- | --------- |
| `--decibans-thresh <DB>`  | `[stages.<s>.gate].decibans_thresh`   | gate      |
| `--cooling-target-iters`  | `cooling_target_iters`                | IF2       |
| `--tempering <B1,B2,...>` | `tempering` (first value must be 1.0) | PGAS      |
| `--max-tree-depth <N>`    | `max_tree_depth`                      | PGAS/NUTS |
| `--trajectory-warmup <N>` | `trajectory_warmup`                   | PGAS      |
| `--csmc-sweeps-per-nuts`  | `csmc_sweeps_per_nuts`                | PGAS      |
| `--n-trajectories <N>`    | `n_trajectories`                      | PGAS      |
| `--diagonal-mass`         | `dense_mass = false`                  | PGAS/NUTS |
| `--no-nuts`               | `use_nuts = false`                    | PGAS      |
| `--no-adapt`              | `adapt = false`                       | PMMH/MH   |
| `--adapt-start <N>`       | `adapt_start`                         | PMMH/MH   |
| `--rho <F>`               | `rho` (correlated pseudo-marginal)    | PMMH      |
| `--record-ancestry`       | `record_ancestry = true`              | PFilter   |
| `--record-prequential`    | `record_prequential = true`           | PFilter   |

The boolean overrides are one-way: they can switch a stage off, and switching it
back means editing the TOML.

> **Four flags do not re-key.** `--decibans-thresh`, `--no-dt-check`,
> `--dt-check-strict` and `--dt-check-halvings` are applied after the run is
> looked up in the store, so on an already-completed stage they are silently
> ignored and the prior verdict is served:
>
> ```
> $ camdl fit run fits/01_mle.toml --stage mle --decibans-thresh 0.1
> ── stage: mle (method=if2) ──
>   cache hit — reusing fits/01_mle-2030ba2b/01-mle-fee126b1/seed_1-06cbd6b3
> $ camdl fit summary results/fits/01_mle-2030ba2b
>     decibans leg:    Δ = 0.9 dB / threshold 30.0 dB  ✓  (σ_max=0.02)
> ```
>
> Delete the stage leaf before rerunning with any of them. Every other flag in
> the table above re-keys correctly.

**Seeds and fits.** A fit runs at a single base seed — `--seed N`, default 1 —
and derives each chain's seed from it by a golden-ratio mix,
`base ^ (chain_index · 0x9e3779b97f4a7c15)`, not by addition. For a
start-sensitivity sweep, the `fit.toml` also accepts a top-level `fit_seeds`
list; each entry becomes its own cell and its own leaf:

```toml
fit_seeds = [1, 2]
```

```
━━━ cell 1/2: fit_seed=1 ━━━
   stored mle · …/01-mle-fee126b1/seed_1-06cbd6b3
━━━ cell 2/2: fit_seed=2 ━━━
   stored mle · …/01-mle-fee126b1/seed_2-f69dd668
```

`fit_seeds` is stripped from the fit-level hash, so adding seeds extends a fit
rather than re-keying it.

### 7.3 Sweep Semantics for Fits

`--sweep NAME=SPEC` overrides a parameter in `[fixed]` at each grid point. The
swept parameter must be in `[fixed]`; both failure modes name the fix:

```
$ camdl fit run fits/01_mle.toml --sweep "beta=0.1,0.2"
error: cannot sweep 'beta' — it is in [estimate].
  Sweeps override [fixed] parameters. Move 'beta' to [fixed] first.

$ camdl fit run fits/01_mle.toml --sweep "zzz=0.1,0.2"
error: sweep parameter 'zzz' not found in [fixed].
  Available fixed params: N0, I0, rho, k
```

So a parameter _promotes_ from fixed to swept with no config change:

```toml
[fixed]
rho = 0.6
```

```bash
camdl fit run fits/01_mle.toml --sweep "rho=0.4,0.6"
```

Repeating the flag takes the Cartesian product, and each point runs the full
stage pipeline independently into its own content-addressed fit segment:

```
$ camdl fit run fits/01_mle.toml --sweep "rho=0.5,0.6" --sweep "k=5,10"
sweep: 4 points
═══ sweep point 1/4: rho_0.500__k_5.000 ═══
          best ll=-59.9 (chain 2) in 8.6s
═══ sweep point 2/4: rho_0.500__k_10.000 ═══
          best ll=-59.3 (chain 3) in 8.7s
═══ sweep point 3/4: rho_0.600__k_5.000 ═══
          best ll=-59.6 (chain 3) in 8.8s
═══ sweep point 4/4: rho_0.600__k_10.000 ═══
```

**Convergence gates.** A stage that consumes an upstream stage
(`init_mle = "<stage>"`) is gated on that upstream's convergence before it runs,
and on not regressing below it afterwards. Both gates live on the IF2 stage
kind, so they apply to a `scout → refine` handoff; a PGAS, PMMH or NLopt stage
consuming an upstream MLE is not gated.

At a **single grid point** a gate failure halts the run with exit status 1:

```
$ camdl fit run fits/03_scout_refine.toml
── stage: refine (method=if2) ──
error: refine stage requires scout convergence.

  Scout tail Â (last half of iterations):
    ✗ gamma      Â =  1.145   (> 1.10)
      beta       Â =  0.965

  Scout loglik spread: 0.2 (best chain loglik -58.4)

  Failing: gamma (Â=1.15)

  Pick one:
    - re-run scout with more chains or iterations
    - narrow bounds to the basin scout's best chain found
    - mark weakly-identified initial-state params as `perturb_only_at_t0 = true`
      (reported but not gated)

  To run refine anyway (results may launder multi-modality):
    camdl fit run fit.toml --allow-nonconverged-scout
```

**Under `--sweep` a gate failure does not halt the sweep.** The point is
recorded, its remaining stages are skipped, and the next point runs. The command
exits 0 even when every cell failed, so downstream tooling must read the failure
file rather than the exit status:

```
━━━ sweep summary ━━━
  2 / 2 cells skipped gate
    cell  1 / pt  1 (rho=0.500) stage=refine reason=scout_tail_agreement_gate
    cell  1 / pt  2 (rho=0.600) stage=refine reason=scout_tail_agreement_gate
  details: results/fits/03_scout_refine-a5aae294/sweep_failures.tsv
```

```
$ cat results/fits/03_scout_refine-a5aae294/sweep_failures.tsv
cell	sweep_point	sweep_values	stage	reason
0	0	rho=0.500000	refine	scout_tail_agreement_gate
0	1	rho=0.600000	refine	scout_tail_agreement_gate
```

`reason` is one of `scout_tail_agreement_gate`, `scout_decibans_spread_gate`, or
`regression_gate`. The file is written only when at least one cell failed, so
its absence means every cell cleared its gate. A plotting script should treat a
missing sweep point as "did not run" and a listed one as "ran and failed to
converge".

`--allow-nonconverged-scout` downgrades the pre-stage gate to a warning in both
modes. The post-stage regression gate is not overridable.

The gate reads whatever supplied the stage's starting point, including an
_external_ fit passed as `--init from_mle --mle <leaf>`. On an IF2 stage that
means warm-starting from another fit's MLE also imports that fit's convergence
verdict, and the refusal is phrased in terms of the `scout → refine` handoff
regardless of what the stage is called:

```
$ camdl fit run fits/03_scout_refine.toml --stage scout \
    --init from_mle --mle results/fits/01_mle-2030ba2b/01-mle-fee126b1/seed_1-06cbd6b3
── stage: scout (method=if2) ──
error: refine stage requires scout convergence.
```

Pass `--allow-nonconverged-scout`, or warm-start a Bayesian stage instead, where
no gate is applied.

For a dedicated profile-likelihood workflow — fix a focal parameter, maximise
over the rest at each grid point — use `camdl profile`, which runs IF2 per cell
over its own `--sweep` grid.

---

## 8. fit.toml Examples

The four configs below were all executed end to end. They share one project
layout; every path inside a `fit.toml` resolves **relative to that file**, which
is why the configs in `fits/` reach back out with `../`.

```
models/sir.camdl          # closed SIR, weekly NegBinomial reported cases
data/cases.tsv            # time, weekly_cases
params/sir_fixed.toml     # shared [fixed] block
fits/*.toml
results/                  # output_dir
```

The iteration counts are deliberately small so each example finishes in seconds.
Real fits want one to two orders of magnitude more of everything.

### 8.1 MLE with IF2

The minimum useful fit: one IF2 stage maximising the stochastic-process
likelihood.

```toml
output_dir = "../results"

[model]
camdl = "../models/sir.camdl"

[data.observations]
weekly_cases = "../data/cases.tsv"

[estimate]
beta = { bounds = [0.05, 1.0], start = 0.25 }
gamma = { bounds = [0.01, 0.5], start = 0.10 }

[fixed]
N0 = 10000
I0 = 10
rho = 0.6
k = 10.0

[config]
dt = 1.0

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 200
iterations = 10
cooling = 0.7
```

```
$ camdl fit run fits/01_mle.toml --seed 1
   cached IR for sir.camdl (97c3d4c1)
fit: fits/01_mle.toml (1 stage)
  model:    fits/../models/sir.camdl
  estimate: beta, gamma
  fixed:    N0, I0, rho, k
  output:   …/results/fits/01_mle-2030ba2b

── stage: mle (method=if2) ──
  note: `init = "uniform_unconstrained"` draws every chain's start, so `[estimate].start` is unused here for: beta, gamma. Use `init = "single"` to start every chain at the declared values, or drop the `start` entries.
running 4 chains × 200 particles × 10 iterations, cooling=0.7, dt=1

transforms (chain 1 of 4; chains start at different points — see chain_starts.tsv):
  beta         log     [0.05, 1]  log(0.0934) = -2.37  rw_sd=0.0335 (0.358/step, auto)
  gamma        log     [0.01, 0.5]  log(0.1775) = -1.73  rw_sd=0.0138 (0.078/step, auto)

  ⚠ 2/2 parameters using auto rw_sd. Check traces and set explicit values.

cooling: cf50=0.70, reached at iter 50 (target), over a 10-iteration run × 12 observations
  iter   1: rw_sd at 99.3%
  iter  50: rw_sd at 70.0% (cf50 reached)
  iter  10: rw_sd at 93.1% (run end)

evaluating loglik (every 10 iterations, all 4 chains)...
  chain 1: ll=-58.4      chain 2: ll=-58.4      chain 3: ll=-58.6      chain 4: ll=-58.5

loglik-eval: re-scoring final-iter θ̂ (4 chains × 8 replicates @ 4000 particles)...

best chain: 2 (loglik=-58.38 ± 0.01)
chain clean logliks: [-58.5, -58.4, -58.6, -58.6]

Â:
  beta         Â=0.965 ✓
  gamma        Â=1.145 ~

dt-convergence at θ̂: PASS  (|Δ_leg1| = 0.65, |Δ_leg2| = 1.16 nats (vs τ = 2.00); converged.)

── diagnostics ──────────────────────────────────────
  i Auto rw_sd for 'beta': 0.033493.
  i Auto rw_sd for 'gamma': 0.013831.
  ! 1/2 parameters have Â > 1.1 (max 1.15).
  0 error(s), 1 warning(s), 2 info

   stored mle · fits/../results/fits/01_mle-2030ba2b/01-mle-fee126b1/seed_1-06cbd6b3/
          best ll=-58.4 (chain 2) in 8.5s
```

Two things the run says that are worth reading. The `note` about `start`: the
default `init = "uniform_unconstrained"` draws each chain's start, so the
`start` values in `[estimate]` are inert unless you also set `init = "single"`.
And the `Â` block: chain agreement on `gamma` is 1.145, which at ten iterations
means the chains have not agreed on a basin. `camdl fit summary` renders the
same run as a verdict:

```
$ camdl fit summary results/fits/01_mle-2030ba2b
results/fits/01_mle-2030ba2b/
  camdl 0.1.0+3e2b2888

══ mle ═══════════════════════════════════════════════════════════════════════
  best loglik:  -58.4 (if2)  (loglik-eval, max across chains)
  chains:       4

  compound scout-convergence gate
    Â leg:           max Â = 1.145 (gamma)  ✗  (threshold 1.01)
    decibans leg:    Δ = 0.9 dB / threshold 30.0 dB  ✓  (σ_max=0.02)
    overall:         ✗ FAIL

  parameter estimates (loglik-eval, selected chain θ̂)
    beta         = 0.458330      Â=0.965 ✓
    gamma        = 0.263048      Â=1.145 ✗

  per-chain loglik-eval (4 chains)
    chain        loglik     ± se
         1       -58.51   ± 0.02
         2       -58.38   ± 0.01  ← selected
         3       -58.57   ± 0.01
         4       -58.59   ± 0.01

  ESS at θ̂
    min  =     1772  (at obs step 1)    mean =     3158

  dt-convergence at θ̂ (Richardson)
    verdict: PASS    (|Δ_leg1| = 0.65, |Δ_leg2| = 1.16 nats (vs τ = 2.00); converged.)

  provenance
    final_params.toml ↔ mle_params.toml: ✓ params match
    fit_state.toml ↔ final_params.toml:   ✓ params match
```

The gate verdict here is informational — nothing consumes this stage. It becomes
binding the moment a downstream stage declares `init_mle = "mle"` (§8.2).

### 8.2 MLE, posterior sampling, and a filter evaluation

A three-stage pipeline: IF2 finds the basin, PGAS samples the posterior
warm-started from it, and a particle filter re-scores θ̂ with a replicate spread.

Priors are **required** for a Bayesian stage. They are externally-tagged inline
tables keyed by the distribution name —
`prior = { log_normal = { mu = …, sigma
= … } }`, not
`prior = { dist = "log_normal", … }`. A `fit.toml` prior overrides the model's
`~ <dist>(...)` declaration; if neither is present, camdl warns and falls back
to flat, and `prior = { flat = {} }` is how you ask for flat on purpose without
the warning.

```toml
output_dir = "../results"

[provenance]
derived_from = "01_mle.toml"
reason = "add a Bayesian stage; scout finds the basin, PGAS samples it"

[model]
camdl = "../models/sir.camdl"

[data.observations]
weekly_cases = "../data/cases.tsv"

[estimate]
beta = { bounds = [0.05, 1.0], prior = { log_normal = { mu = -1.0, sigma = 0.5 } } }
gamma = { bounds = [0.01, 0.5], prior = { log_normal = { mu = -2.0, sigma = 0.5 } } }

[fixed]
N0 = 10000
I0 = 10
rho = 0.6
k = 10.0

[config]
dt = 1.0

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 200
iterations = 10
cooling = 0.7

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 50
sweeps = 200
burn_in = 50
init_mle = "scout"

[stages.evaluate]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 500
replicates = 5
init_mle = "scout"
```

TOML ordering bites here: `output_dir` is a top-level key, so it must appear
_before_ the first `[table]` header. Written after `[provenance]` it is parsed
as a provenance field and rejected:

```
error: error in fits/02_posterior.toml:
parse error: TOML parse error at line 5, column 1
  |
5 | output_dir = "../results"
  | ^^^^^^^^^^
unknown field `output_dir`, expected `derived_from` or `reason`
```

The PGAS stage runs NUTS on θ given the conditioned trajectory:

```
── stage: posterior (method=pgas) ──
  NUTS enabled (gradient expressions found in IR)
  dense mass matrix estimated (sweep 35):
    beta         sd=0.017766
    gamma        sd=0.052177
    correlations: bet-gam=0.42
  NUTS fully adapted (sweep 50):
    final step_size: 0.172352
  chain 1 done: 2.7s, acceptance: [beta=92%, gamma=92%]
  chain 2 done: 2.7s, acceptance: [beta=90%, gamma=90%]

Rhat / ESS:
  beta         Rhat=2.065 ✗ ESS=NaN
  gamma        Rhat=1.675 ✗ ESS=NaN
  draws.tsv: 60 posterior samples (all 6 params)

── diagnostics ──────────────────────────────────────
  x parameter acceptance rate 92.5% is outside healthy range [15%, 50%].
  x Rhat for 'beta' is 2.065 (threshold 1.1). Chain estimates have not converged.
  6 error(s), 0 warning(s), 0 info

── stage: evaluate (method=pfilter) ──
  pfilter rep 1/5: loglik=-58.3
  pfilter rep 2/5: loglik=-58.3
  pfilter rep 3/5: loglik=-58.5
  pfilter rep 4/5: loglik=-58.3
  pfilter rep 5/5: loglik=-58.4

  loglik = -58.4 ± 0.1 (5 reps, 500 particles, 0.2s)
  prequential: elpd=-58.30, mean_crps=316.328, PIT 90% cov=0.08
```

R̂ of 2.07 at 200 sweeps is a not-converged posterior, reported as errors rather
than buried — a real run needs thousands of sweeps. The `pfilter` stage writes
`prequential.tsv`, which is what `camdl compare` reads to score fits against
each other.

Once a Bayesian stage exists, `fit predict` replays the posterior forward and
writes the predicted-vs-observed pair:

```
$ camdl fit predict fits/02_posterior.toml --n-draws 20
fit predict: free_forward horizon — subsampling 20 of 60 posterior draws (raise with --n-draws)
fit predict: one_step horizon — subsampling 20 of 60 posterior draws (raise with --n-draws)
fit predict: horizon=free_forward+one_step(20 draws) treatment=posterior, 1 scenario(s) [fitted], 1 stream(s), 60 draws from pgas stage 'posterior'
wrote fits/../results/fits/02_posterior-77595169/predictive/weekly_cases.tsv
wrote fits/../results/fits/02_posterior-77595169/predictive.json
wrote fits/../results/fits/02_posterior-77595169/observed/weekly_cases.tsv
wrote fits/../results/fits/02_posterior-77595169/observed.json
```

```
$ head -3 results/fits/02_posterior-77595169/predictive/weekly_cases.tsv
scenario	time	horizon	treatment	rhat_max	ess_min	n_draws	q05	q25	q50	q75	q95
fitted	0	free_forward	posterior	2.0652		20	0	0	0	0	0
fitted	7	free_forward	posterior	2.0652		20	10.7	19	38	48.25	98.4
```

> **Current limitation.** A fit whose stage list contains a `pfilter` stage is
> dropped from `camdl fit table` with a `unknown fit-stage method 'pfilter'`
> warning, and `fit summary --format json` emits an empty `table_row` for it.
> The text `fit summary` still renders the IF2 and PGAS stanzas correctly. If
> you rely on the cross-fit table, keep the filter evaluation in a separate
> `fit.toml`.

### 8.3 Deterministic ODE backend

The same model, fitted against the ODE skeleton rather than the stochastic
process: an NLopt optimiser for the MLE, then gradient-free Metropolis-Hastings
for the posterior. `camdl fit methods` lists every supported (algorithm,
backend) pair; an invalid pair is rejected at load.

This is a **different statistical object** — `p(y | θ, ODE skeleton)`, not
`p(y | θ)` — appropriate for equilibrium or large-population models where the
particle filter is structurally redundant. The runtime says so at every stage:

```
ℹ nl-sbplx (Subspace simplex; robust to boundary non-smoothness): deterministic MLE on the ODE-skeleton likelihood.
  camdl computes p(y|θ, ODE_skeleton) under nl-sbplx, not the stochastic-process p(y|θ) IF2/PGAS/PMMH compute. In low-noise regimes the two converge empirically; verify rather than assume.
```

```toml
output_dir = "../results"

[model]
camdl = "../models/sir.camdl"

[data.observations]
weekly_cases = "../data/cases.tsv"

[estimate]
beta = { bounds = [0.05, 1.0], prior = { log_normal = { mu = -1.0, sigma = 0.5 } } }
gamma = { bounds = [0.01, 0.5], prior = { log_normal = { mu = -2.0, sigma = 0.5 } } }

[fixed]
N0 = 10000
I0 = 10
rho = 0.6
k = 10.0

[config]
dt = 0.5

[stages.mle]
algorithm = "nl-sbplx"
backend = "ode"
chains = 3
max_evals = 200

[stages.posterior]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 400
burn_in = 100
init_mle = "mle"
```

```
$ camdl fit run fits/05_ode.toml --seed 1
── stage: mle (method=nl-sbplx) ──
⚠ Phase 1 typhoid validation passed; other model classes still gathering downstream feedback.

  status: 1 converged, 2 max-eval, 0 failed (of 3)
  chain-agreement: rel range = 49.10% bound | abs range = 3.698e-1   ✗
  loglik-eval:     Δ = 0.0 dB / threshold 30 dB                ✓
  wall: 0.11s (3 chains)

dt-convergence at θ̂: PASS  (|Δ_leg1| = 0.00, |Δ_leg2| = 0.00 nats (vs τ = 0.50); converged.)

── stage: posterior (method=mh) ──

ODE marginal-likelihood check at base θ (deterministic)...
  ODE log L = -57.8 (deterministic; no PF variance)
  observations (1 stream):
    ✓ weekly_cases     incidence(infection)         NegBinomial

acceptance rates:
  chain 1: 51.3% ~ high
  chain 2: 51.7% ~ high

Rhat:
  beta         Rhat=1.105 ~ ESS=NaN
  gamma        Rhat=1.051 ✓ ESS=18
```

`nuts` is the gradient-based sibling of `mh` on this backend (`warmup` +
`samples` instead of `iterations` + `burn_in`). It requires a differentiable
model: the capability gate refuses an unsupported rate or observation gradient,
an adaptive `rk45` integrator, a scheduled effect, or an initial condition the
gradient path cannot seed.

### 8.4 Shared fixed parameters and a train/holdout split

`[fixed] from_file` bulk-loads a params TOML that several fits share; inline
`[fixed]` entries override individual keys from it. Unlike every other path in
the file, **`from_file` resolves relative to the working directory**, not to the
`fit.toml` — write it as the path you would type from where you run `camdl`.

```toml
# params/sir_fixed.toml
N0 = 10000
I0 = 10
rho = 0.6
k = 10.0
```

Out-of-sample validation is done by splitting the data file, not by a key in the
config. `camdl data split` writes the two files:

```
$ camdl data split data/cases.tsv --at-time 56 \
    --train data/train.tsv --holdout data/holdout.tsv
Split at t = 56 (column 'time')
  Train:   9 observations, t ∈ [0, 56]
  Holdout: 3 observations, t ∈ [63, 77]
  Written: data/train.tsv, data/holdout.tsv
```

The fit then points at the training file:

```toml
output_dir = "../results"

[model]
camdl = "../models/sir.camdl"

[data.observations]
weekly_cases = "../data/train.tsv"

[estimate]
beta = { bounds = [0.05, 1.0] }
gamma = { bounds = [0.01, 0.5] }

[fixed]
from_file = "params/sir_fixed.toml"

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
```

```
$ camdl fit run fits/08_holdoutfiles.toml --seed 1
cooling: cf50=0.70, reached at iter 50 (target), over a 5-iteration run × 9 observations
best chain: 1 (loglik=-48.01 ± 0.01)
   stored scout · …/results/fits/08_holdoutfiles-812b33c5/01-scout-c3b8cbc0/seed_1-06cbd6b3
```

Nine observations, as the split promised. `fit summary --params-only` emits θ̂ as
a flat params TOML, which `camdl pfilter` reads directly, so the held-out window
is scored in two commands:

```
$ camdl fit summary results/fits/08_holdoutfiles-812b33c5 --params-only
# camdl fit summary --params-only --stage scout
# source: results/fits/08_holdoutfiles-812b33c5/01-scout-c3b8cbc0/seed_1-06cbd6b3/final_params.toml
# camdl: 0.1.0+3e2b2888

I0 = 10.0
N0 = 10000.0
beta = 0.4115404138
gamma = 0.1802936713
k = 10.0
rho = 0.6

$ camdl pfilter models/sir.camdl --params theta.toml \
    --data weekly_cases=data/holdout.tsv --particles 500
pfilter: 3 observations × 1 streams, 500 particles, dt=1, seed=1
pfilter: bound streams: weekly_cases(neg_binomial)
-43.9094
```

Read that number carefully. The filter starts at the model's `simulate { from }`
and the first held-out observation is at t = 63, so the first scored bin
accumulates incidence over the entire 0–63 warm-up rather than over one weekly
cadence. `condition_from` fixes exactly this, and `pfilter` applies it the same
way `fit run` does: `--condition-from` if given, else the `condition_from` of
the toml passed to `--fit`. A `pfilter` that scored a window the fit never
scores would report a log-likelihood incomparable with the fit's, so the two
share one code path (`apply_conditioning_windows`), and W329 — the
wide-first-window enforcer — runs here too.

`camdl fit predict` plus `camdl compare` answers a different question, replaying
the fitted trajectory forward into the held-out window rather than re-filtering
it; reach for that when you want a predictive score rather than a filter
log-likelihood.

> **Current limitation.** `[data] holdout_after = <t>` and the `[data.holdout]`
> file block are accepted by the parser and folded into the fit hash, but no
> stage reads them: a fit declaring `holdout_after` trains on the **full**
> series, with the same θ̂ and the same log-likelihood as one that omits the key.
> Split the file instead.

### 8.5 Comparing and deriving configs

Two read-side commands make a family of fits navigable.

`fit diff` reports what changed between two configs, in fit-semantic terms
rather than as a text diff:

```
$ camdl fit diff fits/01_mle.toml fits/02_posterior.toml
diff: fits/01_mle.toml → fits/02_posterior.toml

  beta: prior (none) → log_normal(mu=-1, sigma=0.5)
  gamma: prior (none) → log_normal(mu=-2, sigma=0.5)

Stages:
  stage 'evaluate': (new) pfilter
  stage 'posterior': (new) pgas
  stage 'scout': (new) if2
  stage 'mle': (removed)
```

`fit new` copies a config, stamps a `[provenance]` block onto it, and points at
the source fit's results so the new one can warm-start from them:

```
$ camdl fit new --from fits/01_mle.toml fits/04_derived.toml
  [provenance] derived_from = "fits/01_mle.toml"
  hint: set starts_from on your first stage to the last stage leaf under fits/../results/fits/01_mle-2030ba2b
        (run `camdl list` to find the exact stage-leaf path)
created fits/04_derived.toml
```

The key that hint means is `init_mle` (`starts_from` is the retired spelling and
is rejected at load).

`fit table` renders every fit under a results root, one row each:

```
$ camdl fit table results/fits
fit_id     label                  stem           method   stages converged     best_ll ll_type          age
--------------------------------------------------------------------------------------------------------------
649b2ecc   <unlabelled>           09_pgas_only   pgas     poste… no                  — complete_data     4m
2030ba2b   auto rw_sd, take 1     01_mle         if2      mle    no              -58.4 if2              11m
812b33c5   <unlabelled>           08_holdoutfil… if2      scout  no              -48.0 if2              13m
a5aae294   <unlabelled>           03_scout_refi… if2      scout… no              -58.4 if2              23m
8e6e8964   <unlabelled>           05_ode         mh       mle+p… no              -58.9 marginal         23m
```

Labels come from `--label` at run time or from the top-level `camdl label`
afterwards:

```
$ camdl label 2030ba2b "auto rw_sd, take 1"
ok: label set to "auto rw_sd, take 1" on ./results/fits/01_mle-2030ba2b
```

Filters: `--converged` / `--gate-failed`, `--with-stage <name>`,
`--with-method if2|pgas|pmmh`, `--model <hash-prefix>`, `--hash <hash-prefix>`,
`--since-seconds <n>`, `--label-pattern <glob>`. Formats: `text` (default),
`json`, `md`, `csv`. `--quantity <name>` adds a column carrying the posterior
median of a scalar generated quantity, deriving it on demand for a fit that has
not been predicted yet.

Note that `fit_id` is the **fit-level** hash — model, data, `[estimate]`,
`[fixed]` — computed with `stages`, `fit_seeds` and `output_dir` deliberately
excluded. Two configs that differ only in their stage list therefore share a
`fit_id`, and `--hash <prefix>` can return more than one row; the directory stem
is what separates them.

## 9. Provenance and Cache Invalidation

### 9.1 The identity model

Every artifact camdl writes — a forward trajectory, a fit stage, a particle
filter evaluation, a survey, a profile grid point, a multi-cell ensemble — is
stored as a **content-addressed leaf** in one store rooted at the output
directory (default `results/`, see `run_paths::DEFAULT_OUTPUT_ROOT`). There is
no second scheme: fits are content-addressed exactly like simulations. The only
thing that is "named" about a fit directory is the human-readable _label_ in
each path segment, which is provenance and never identity.

A leaf's identity is **not** one flat hash. It is an ordered tuple of per-level
content hashes, and the leaf address is derived from that tuple:

```
run_id = SHA256( HASH_VERSION:u16 ++ kind_tag:u32 ++ n_levels:u64 ++ level_hash[0..n] )
```

(`runid::kind::run_id`, `rust/crates/runid/src/kind.rs:79`). `kind_tag` is the
`ArtifactKind` declaration index (`Sim`=0, `FitStage`=1, `Pfilter`=2,
`Survey`=3, `ProfilePoint`=4, `Obs`=5, `Projection`=6, `SimEnsemble`=7), so two
kinds with a coincidentally-equal level sequence cannot alias.

Each level hash is the `ContentAddressed::content_hash` of a resolved input
slice, itself `SHA256(HASH_VERSION ++ canonical_bytes)`. The store path is a
readable _factoring_ of that tuple: one directory segment per level, each
`{label}-{hash8}`, where `hash8` is the first 4 bytes of the level hash as 8 hex
characters and the label is cosmetic (`runid::layout`). Navigation and every
read path resolve through `run.json`, never through path segments.

Two rules govern what may enter a hash (`rust/crates/runid/src/lib.rs`, "Adding
a field to identity"):

- A field that **changes the stored bytes is identity** and must re-key.
- A pure **re-encoding of the same values is presentation** and is stripped.

A missed re-key silently serves a stale result for different inputs. That is the
worst failure this subsystem can produce, and it is worse than an
over-invalidation.

### 9.2 What enters a run's identity

#### 9.2.1 The canonical hasher

One pinned function and encoding, in `rust/crates/runid/src/hash.rs`:

- SHA-256, recorded by `HASH_VERSION` (currently `1`), folded into every root
  hash so the whole store migrates on a single bump.
- **Domain separation.** Every named type writes its fully-qualified type name
  (length-prefixed) then its `SCHEMA_VERSION` before any field.
- **Length prefixing.** Every variable-length value (string, byte slice, `Vec`,
  map, set) writes a `u64` LE element count first.
- **Primitives.** Fixed-width little-endian integers; `bool` as one byte; `char`
  as `u32`.
- **Floats, two policies.** Resolved user inputs go through `FiniteF64`, which
  rejects `NaN`/`±Inf` at construction and normalizes `-0.0 → +0.0`. Structural
  IR floats are hashed as raw `f64::to_bits`, keeping `±0.0` and NaN payloads
  distinct to match the IR's own `ConstExpr::PartialEq`. Bare `f64` is
  deliberately **not** `ContentAddressed`, so a field cannot silently pick up
  the wrong policy.
- **Maps and sets** iterate in sorted key order, count-prefixed. **`Option`** is
  a `0`/`1` tag byte then payload. **Enums** write the variant index as `u32` LE
  in declaration order, then the payload.

#### 9.2.2 Include-by-default, with one explicit escape

`#[derive(RunInput)]` (`rust/crates/runid-derive/src/lib.rs`) folds **every**
field of a struct in declaration order. There is no skip-if-default: adding a
field re-keys every existing leaf of that kind even at its default value, and
that turnover is intended, not a bug to engineer around. Two attributes exist:

- `#[run_input(provenance)]` on a field — excluded entirely (recorded in
  `run.json`, never hashed).
- `#[run_input(schema_version = N)]` on the type — bumps only that type's key.

A field whose type is not `ContentAddressed` is a compile error, so an input
cannot be made unhashable by accident.

The **fit** level takes the other path: `FitConfigV2` is serde-only and enters
identity as the digest of its key-sorted canonical JSON, so
`#[serde(skip_serializing_if = …)]` controls hash membership there — a
default-skipped field stays out of the hash.

#### 9.2.3 The model digest

`ModelDigest::from_model` (`rust/crates/runid/src/inputs.rs:194`) is the single
constructor every identity path uses:

| field        | value                                                |
| ------------ | ---------------------------------------------------- |
| `ir`         | `model_ir_hash(model)` — the whole canonical IR      |
| `ir_version` | the `ir/VERSION` string (e.g. `"0.30"`)              |
| `engine`     | `EngineVersion` = `"{CARGO_PKG_VERSION}+{git-hash}"` |

`model_ir_hash` applies `normalize_for_hash`, which blanks the two
pure-presentation IR fields — `output.format` and `simulation.time_semantics` —
before hashing, so a rendering choice never re-keys a run. The normalization
lives inside the constructor, not at any call site, so no artifact kind can opt
out of it.

The IR walk is hand-written (`rust/crates/runid/src/ir_hash.rs`, schema version
`SV = 2`). `Model::hash_into` (`ir_hash.rs:1117`) folds, in order: `name`,
`version`, `time_unit`, `description`, `origin`, `origin_rata_die`,
`compartments`, `transitions`, `ode_equations`, `time_functions`, `tables`,
`interventions`, `observations`, `parameters`, `bindings`, `per_eval_bindings`,
`initial_conditions`, `output`, `simulation`, `presets`, `model_structure`,
`balance`, `identity_tracked_compartments`.

Three `ir::Model` fields are deliberately **not** hashed:

- `ic_grad` and every compiler-derived gradient map (`rate_grad`,
  `rate_state_grad`, `sigma_sq_grad`, `projection_state_grad`, each obs
  `Diffable`'s `grad`/`proj_grad`) — deterministic autodiff of already-hashed
  rates over already-hashed parameters, so model identity is
  gradient-independent and a `--no-state-grad` compile keys the same.
- `quantities` and `contrasts` — reporting-only reductions written to a
  regenerated sidecar outside the leaf (§10.7), never to the leaf itself.

One conditional inside `SimulationConfig::hash_into` (`ir_hash.rs:1046`): the
integrator is folded **only when non-default**, i.e. only for
`Integrator::Rk45 { atol, rtol }`. A model declaring the default RK4 hashes as
if the field did not exist.

#### 9.2.4 The levels, per artifact kind

| kind (`store_dir`)           | levels, in path order                     | resolver                         |
| ---------------------------- | ----------------------------------------- | -------------------------------- |
| `Sim` (`sims/`)              | model · config · params · scenario · seed | `cli/src/resolve.rs:179`         |
| `SimEnsemble` (`ensembles/`) | model · config · params · grid            | `cli/src/sim_ensemble_cas.rs:88` |
| `FitStage` (`fits/`)         | fit · stage · seed                        | `cli/src/fit/cas.rs:420`         |
| `Pfilter` (`pfilters/`)      | model · config · params · seed            | `cli/src/pfilter_cas.rs:65`      |
| `Survey` (`surveys/`)        | model · config · box · seed               | `cli/src/survey_cas.rs:61`       |
| `ProfilePoint` (`profiles/`) | profile · point · stage · seed · start    | `cli/src/profile_cas.rs`         |

**`Sim` levels** (`resolve::resolve_trajectory`; the level types are in
`runid::inputs`):

- `model` — `ModelDigest` (§9.2.3). Label: the model file stem.
- `config` — `SimConfig`, `schema_version = 3`: `backend`
  (gillespie/chain_binomial/ode), `dt`, `t_start`, `t_end`, the resolved output
  schedule (`Regular { start, step }` or `AtTimes([…])`), `calendar` (a
  placeholder always `Numeric` today), `allow_degenerate_rates`, `no_flows`, and
  the `columns` allow-list as a sorted set. Label: `{backend}-dt{dt}`.
- `params` — `ResolvedParams`: the resolved base parameter map
  (`BTreeMap<ParamId, FiniteF64>`, i.e. model defaults overlaid by `--params`
  then `--param`, plus any sweep point) and the SHA-256 **content** digests of
  every `--table NAME=FILE` file, sorted by table name. Files are hashed by
  bytes, never by path. Label: `base`, a `k=v` join of the sweep point, or
  `draws` when a full draw vector would overflow the path component.
- `scenario` — `ResolvedScenario`: the sorted `enable` and `disable`
  intervention-id sets and the parameter patch. Only the _delta_; renaming a
  scenario preserves the hash. Label: the scenario name.
- `seed` — `Seed`. Only `process_seed` is hashed; `base_seed` is
  `#[run_input(provenance)]`. Hashing the derived per-cell seed rather than the
  user's `--seed` is what stops a lone run and a sweep cell from aliasing.
  Label: `seed_{n}` — note this renders the _process_ seed on the simulate/batch
  path, so a multi-replicate run shows segments like `seed_5871781006564002452`.

**`FitStage` levels** (`fit::cas::resolve_fit_stage`):

- `fit` — `FitDigest`: the `ModelDigest`; the content digest of each resolved
  _training_ stream; the content digest of each `[data.holdout]` stream; the
  digest of the whole canonicalized fit.toml **less** `stages`, `fit_seeds` and
  `output_dir`; and the engine version. Excluding `stages` is what lets editing
  the posterior block leave the scout leaf reusable. Model and data _paths_ stay
  in the blob — a rename over-invalidates harmlessly.
- `stage` — `StageLevel` = `StageConfig` folded with `deps`. `StageConfig`
  carries the digest of `Stage::identity_payload()` re-augmented with
  `n_trajectories`, plus `target_length` and the **resolved** (not requested)
  observation-time alignment (`Exact` vs `Snap`). Folding `deps` here is what
  makes `02-posterior` re-key when `01-scout` changes.
- `seed` — the resolved fit RNG seed.

`Stage::identity_payload()` deliberately omits the _extension dimension_ (PGAS
sweeps, PMMH iterations) so `--resume` can lengthen a chain without re-keying;
the resume length is keyed separately as `target_length`.

**`SimEnsemble` levels**: the `config` level is only `{backend, dt}`; every
other config knob reaches the ensemble through the `grid` level, which digests
`{n_cells, [(scenario_label, process_seed, draw_idx, sim_run_id)] sorted}`. The
cell **count** is in the key explicitly.

**`Pfilter` `config`** digests
`{particles, replicates, dt, obs_block,
flow_indices, data}` where `data` is the
sorted per-stream file digests; `params` digests the sorted scored point.
Scenario selection reaches identity through the model digest, because the
pfilter path hashes the _resolver's_ model (scenario already applied), unlike
`Sim`.

**`Survey` `config`** digests
`{eval_method, eval_particles, eval_replicates,
data, fixed, scenario}`; `box`
digests `{bounds, n_points}`.

**Lineage.** `deps` is a set of `ArtifactRef { run_id, kind, artifact, digest }`
hashed **sorted by `run_id`** (the one documented exception to the
order-sensitive `Vec` rule), with `kind` provenance-only. Folding the consumed
file's digest as well as the producer's `run_id` means a regenerated upstream
re-keys the consumer while a change to an unrelated sibling artifact does not.

#### 9.2.5 Non-finite floats cannot collide

`serde_json` renders `NaN`/`±Inf` as `null` in both `to_value` and the text
serializer without erroring, so two configs differing only in a non-finite float
would hash equal. Every JSON-digested identity blob is therefore passed through
`fit::cas::ensure_finite` first, a `Serializer` that produces nothing and fails
on any non-finite float anywhere in the value. On the typed path the same job is
done by `FiniteF64`'s constructor.

### 9.3 Store layout and cache semantics

```
<root>/
  index.json                       # derived, rebuildable (§9.6)
  .staging/                        # in-flight commits
  .quarantine/                     # unidentifiable debris moved aside
  sims/
    {model_stem}-{h8}/
      {backend}-dt{dt}-{h8}/
        {param_label}-{h8}/
          {scenario}-{h8}/
            seed_{n}-{h8}/
              run.json             # RunRecord, kind = sim
              traj.tsv
              event_log.tsv        # only with --event-log
              reactive_log.tsv     # only when a reactive policy fired
              obs/                 # declared child, one dir per (obs model, obs seed)
                {obs_hash8}-{obs_seed}/
                  <stream>.tsv
                  obs.json         # NOT a RunRecord
  ensembles/
    {model_stem}-{h8}/{backend}-dt{dt}-{h8}/{param_label}-{h8}/cells-n{N}-{h8}/
      run.json
      ensemble.tsv
  fits/
    {fit_stem}-{h8}/               # the fit level; also holds fit-wide sidecars
      fit.meta.json
      fit.toml.original
      model.camdl.original
      model.ir.json
      model.graph.json
      model.render.json
      {NN}-{stage}-{h8}/
        seed_{n}-{h8}/
          run.json                 # RunRecord, kind = fit_stage
          … stage artifacts (§10.6)
  pfilters/ surveys/ profiles/     # same {label}-{h8} factoring per §9.2.4
```

A path component that would exceed `NAME_MAX` is truncated to 200 bytes and
suffixed `..{hash16}` of the full label; identity is unaffected because the
label is never hashed.

**Lookup** (`runid::store::FsCasStore::lookup`) is tiered and returns one of:

| outcome              | meaning                                                                |
| -------------------- | ---------------------------------------------------------------------- |
| `Hit(record)`        | identity matches, `Completed`, exact-set integrity ok                  |
| `Miss`               | no `run.json` at the path                                              |
| `Stale(Incomplete)`  | `status != Completed` (crashed or in-flight)                           |
| `Stale(Corrupt)`     | unparseable `run.json`, or a listed file missing / wrong size or mtime |
| `Stale(OrphanFiles)` | an unlisted, undeclared file or subdirectory is present                |
| `Stale(SchemaDrift)` | `hash_version` or `format_version` is not current                      |
| `Collision(record)`  | a _different_ full identity occupies this path (short-hash clash)      |

A `Collision` is never touched: the writer escalates the final segment (`{seg}`
→ `{seg}~{hash16}` → `{seg}~{full64}`) until it finds a free candidate, and the
reader enumerates sibling directories rather than reconstructing names.

The integrity gate is `bytes + mtime` per manifest entry. It is a _cheap_ gate —
`cp` without `-p`, an unzip, or a checkout will change mtimes and make every
leaf read `Stale`, and conversely a file edited in place to the same length with
the mtime restored passes. The strong `digest` recorded per artifact is **not**
recomputed on any read path today.

**What a cache hit actually saves.**

- `camdl batch run` skips the simulation itself: `CasSink::should_run` returns
  `false` on a `Hit` and the cell is never computed.
- `camdl simulate` **always recomputes every cell** — `SimSink` deliberately
  does not override `should_run`, because the combined mirror needs every cell's
  rows. What the hit saves is the write: `commit_atomic` sees an
  `AlreadyCompleted` leaf and discards the freshly staged bytes.

Consequences of that second rule, which are load-bearing for anyone reasoning
about the store:

- On a hit the leaf keeps the **bytes of the first writer**. A second run whose
  identity matches but whose bytes differ leaves no trace.
- `simulate --force` re-runs the simulation but does **not** overwrite an
  existing same-identity leaf. Replacing a leaf's contents requires deleting the
  directory.
- An artifact that only some invocations produce (`event_log.tsv`) cannot be
  added to a leaf that already exists without a fresh identity or a manual
  delete; call sites check for the file's existence rather than assuming it.
- `--label` is the one recorded field refreshed on a hit: the leaf's
  `run.json.provenance.label` is rewritten in place when it differs.

**What re-keys a `Sim` run** (verified by perturbing one input at a time against
`ocaml/golden/sir_basic.camdl`):

| change                                | re-keys? | level        |
| ------------------------------------- | -------- | ------------ |
| model file edit                       | yes      | model        |
| camdl (engine) version                | yes      | model        |
| `ir/VERSION`                          | yes      | model        |
| `--backend`                           | yes      | config       |
| `--dt`                                | yes      | config       |
| `--output-every`                      | yes      | model[^oe]   |
| `--no-flows`, `--columns`             | yes      | config       |
| `--allow-degenerate-rates`            | yes      | config       |
| `--param`, `--params`                 | yes      | params       |
| `--table` file _contents_             | yes      | params       |
| `--scenario`, `--enable`, `--disable` | yes      | scenario     |
| `--seed`, `--seeds`, `--replicates`   | yes      | seed         |
| `--emit-every`                        | no[^ee]  | obs child    |
| `--label`                             | no       | provenance   |
| `-o`, `--obs*`, `--draws-out`         | no       | mirrors only |
| `--integrator`                        | **no**   | see §9.8     |
| `--dates`                             | **no**   | see §9.8     |

[^oe]: `--output-every` re-keys through the _model_ level, not `config`:
    `util::rematerialize_with_output_every` rewrites the compiled IR to a temp
    file that both the engine and the identity resolver then load. This is the
    pattern a model-mutating CLI flag must follow **when the flag changes the
    trajectory**.

[^ee]: `--emit-every` deliberately does NOT follow that pattern. It changes only
    the emitted observations, so it keys the `obs/` child (`obs_hash`, §10.5)
    and leaves the sim `run_id` alone — two cadences are two obs subtrees under
    one trajectory leaf. Lowering it into the IR would move the model hash a
    `fit` against real data shares, re-keying a fit over a change
    `emit_schedule` cannot make to a likelihood.

### 9.4 Fit identity and re-running a changed fit

There is no staleness _error_. A fit whose config changed simply resolves to a
different `run_id`, lands in a different leaf, and runs; the previous leaf stays
addressable. There is no `provenance.json`, no stored `config_hash` compared on
re-run, and no `--force` prompt on the fit path.

What each edit does:

| edit                                                                                    | effect                                               |
| --------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| model, training data bytes, holdout data bytes                                          | new `fit` level → every stage re-runs                |
| any fit.toml field outside `stages`/`fit_seeds`/`output_dir`                            | new `fit` level → every stage re-runs                |
| a `[stages.X]` field                                                                    | new `stage` level for X only; upstream stages reused |
| a stage's upstream (`starts_from`, `--mle`, `--survey-path`, `--posterior`, `--params`) | new `deps` → new `stage` level downstream            |
| `fit_seeds`                                                                             | new `seed` level                                     |
| `output_dir`                                                                            | nothing (write location is provenance)               |

Chain-start _sources_ are keyed by content, not path: `--posterior`'s
`draws.tsv` and `--params`' TOML each enter `deps` as their file digest, so
rewriting the file in place re-keys the fit.

**A separate, narrower hash guards `--resume`.**
`fit::provenance::fit_stage_hash` digests
`model_ir_json ++ data bytes ++ [estimate] specs ++ resolved [fixed] ++
simplex groups ++ stage_name ++ Stage::identity_payload() ++ seed ++
camdl version`,
and is stored in each chain's `resume_state.bin`. PMMH and PGAS refuse to resume
when it differs:

```
error: config hash mismatch for chain 1 — model/data/priors have changed
since the original run. Cannot resume. Re-run from scratch with --force.
```

This hash is **not** the CAS fit identity and covers a strictly narrower input
set (no `[data.holdout]` contents, no `[config]` block, no resolved
`obs_alignment`, no `n_trajectories`, and no presentation normalization of the
model IR).

### 9.5 run.json — the per-run metadata contract

`run.json` is the API between camdl and everything that reads its output
(`camdl list/show/cat`, viewers, notebooks). The on-disk schema is
`runid::record::RunRecord` (`rust/crates/runid/src/record.rs:159`);
`FORMAT_VERSION` is `1`. **The record itself is never hashed** — identity comes
only from the levels — which is why declaring an `obs/` child cannot change the
trajectory's `run_id`.

A real `sims/` leaf record (hashes and paths abbreviated):

```json
{
  "format_version": 1,
  "kind": "sim",
  "run_id": "467417eb…0258",
  "hash_version": 1,
  "ir_version": "0.30",
  "engine_version": "0.1.0+<git>",
  "levels": [
    {
      "name": "model",
      "label": "sir_basic",
      "hash": "cd37d79d…32e0",
      "schema_version": 1
    },
    {
      "name": "config",
      "label": "chain_binomial-dt1",
      "hash": "9fe90ef5…3c4f",
      "schema_version": 1
    },
    {
      "name": "params",
      "label": "base",
      "hash": "39391d3f…4ba3",
      "schema_version": 1
    },
    {
      "name": "scenario",
      "label": "baseline",
      "hash": "812a8a17…5b96",
      "schema_version": 1
    },
    {
      "name": "seed",
      "label": "seed_1",
      "hash": "06cbd6b3…c5b4",
      "schema_version": 1
    }
  ],
  "status": "completed",
  "artifacts": {
    "traj.tsv": {
      "bytes": 1537,
      "mtime": "1786506768.083634697",
      "digest": "164a63c1…25f7"
    }
  },
  "output_schema": {
    "traj.tsv": {
      "role": "trajectory",
      "columns": [
        { "name": "t", "role": "time" },
        { "name": "S", "role": "state" },
        { "name": "I", "role": "state" },
        { "name": "R", "role": "state" },
        { "name": "flow_infection", "role": "flow" },
        { "name": "flow_recovery", "role": "flow" }
      ]
    }
  },
  "provenance": {
    "argv": [
      "<camdl>",
      "simulate",
      "<model>.camdl",
      "--scenario",
      "baseline",
      "--seed",
      "1"
    ],
    "created_at": "2026-08-12T03:52:48Z",
    "camdl_version": "0.1.0+<git>",
    "source_paths": ["<model>.camdl"]
  }
}
```

A `fits/` stage leaf adds `deps` (when it consumed an upstream), nests artifact
keys (`chain_1/trace.tsv`), keys per-chain schema entries under a `{n}`
wildcard, and carries a populated `inputs`:

```json
"inputs": {
  "algorithm": { "algorithm": "if2", "backend": "chain_binomial",
                 "chains": 2, "cooling": 0.7, "iterations": 8, "particles": 1000 },
  "backend": "chain_binomial",
  "best_chain": 1,
  "best_loglik": -163.87886531306089,
  "fit_hash": "7aac3a33…a90a",
  "method": "if2",
  "n_chains": 2,
  "seed": 1,
  "stage": "scout",
  "starts_from": null,
  "wall_time_seconds": 25.255000542
}
```

**Fields.**

| field            | type                                       | meaning                                                                                       |
| ---------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `format_version` | u16                                        | `run.json` schema version; a mismatch is `Stale(SchemaDrift)`                                 |
| `kind`           | snake-case `ArtifactKind`                  | `sim`, `fit_stage`, `pfilter`, `survey`, `profile_point`, `obs`, `projection`, `sim_ensemble` |
| `run_id`         | 64-hex                                     | the leaf address; the identity gate on every lookup                                           |
| `hash_version`   | u16                                        | `HASH_VERSION` of the encoding that produced the hashes                                       |
| `ir_version`     | string                                     | `ir/VERSION` at write time (e.g. `"0.30"`)                                                    |
| `engine_version` | string                                     | `"{pkg_version}+{git-hash}"`                                                                  |
| `levels`         | `[{name, label, hash, schema_version}]`    | the factored identity in path order                                                           |
| `deps`           | `[{run_id, kind, artifact, digest}]`       | consumed upstreams; omitted when empty                                                        |
| `status`         | `"running"` \| `"completed"` \| `"failed"` | a plain string, not an object                                                                 |
| `artifacts`      | map path → `{bytes, mtime, digest}`        | EXACT SET over the leaf's own files; `mtime` is `"{secs}.{nanos:09}"`                         |
| `output_schema`  | map path → `{role, columns}`               | declared column roles (§10.8); omitted when empty                                             |
| `children`       | map namespace → `[hash]`                   | declared child sub-artifact dirs (`obs`); omitted when empty                                  |
| `inputs`         | arbitrary JSON                             | display/audit summary, **never hashed**; omitted when null                                    |
| `provenance`     | object                                     | `argv`, `label`, `created_at`, `camdl_version`, `source_paths` — all recorded, never hashed   |

`status` transitions depend on the write mode. `sim` and `sim_ensemble` commit
**atomically**: the leaf appears only once complete, and there is no
discoverable in-flight record. `fit_stage`, `pfilter`, `survey` and
`profile_point` **stream**: they claim the directory, write a `"running"` record
and an exclusive `.lock`, stream output in, then finalize to `"completed"` with
the exact-set manifest and the post-run `inputs`. A stale `.lock` whose PID is
dead is reclaimed; a live one refuses with `fit in progress`.

Three `provenance` fields — `finished_at`, `host`, `thread_count` — exist in the
schema and are never populated by any writer.

Sub-artifacts that are _not_ `RunRecord`s:

- `obs/{obs_hash8}-{obs_seed}/obs.json` —
  `{obs_hash, obs_seed, process_seed, streams, version}`, where `obs_hash` is
  the SHA-256 of the model's serialized `observations` block. The parent
  declares this directory under `children.obs`, but the declared hash is
  `SHA256("{run_id}:{obs_seed}:{obs_hash}")` and appears nowhere inside the
  child.
- `fits/{fit_stem}-{h8}/fit.meta.json` — the fit-wide sidecar:
  `{model_path, model_identity, fit_toml_path, fit_toml_hash, data_hashes,
  estimated, fixed, resolved_priors, parameters_provenance, schema}`,
  where `schema` describes each observation stream (`name`, `index_dims`,
  `value_column`, `value_kind`, `likelihood`).

### 9.6 The derived index (`index.json`)

`<root>/index.json` caches
`run_id → (rel_path, kind, label, status,
created_at)` so prefix resolution in
`show`/`cat` does not re-walk the tree. It is versioned (`INDEX_VERSION = 1`),
written with the same atomic tmp+rename+fsync ordering as `run.json`, and
**never authoritative**:

1. A lookup the index cannot satisfy falls back to a full walk and refreshes the
   index from it, so a leaf written out of band is still found.
2. Every index hit is verified against the live tree — the entry's `run.json` is
   re-read and its `run_id` re-checked. A dead or re-identified entry is dropped
   and the lookup falls through to the walk.

A malformed, absent, or version-mismatched `index.json` is a clean miss, never
an error. `camdl dev reindex [ROOT]` rebuilds it from a fresh walk.

### 9.7 Enumerating a batch — no manifest file

There is no batch-level `manifest.json`. Each completed cell is independently a
content-addressed `run.json` leaf under `sims/` (the system of record). To
enumerate a sweep, walk those leaves or read the derived `index.json` (§9.6).
`camdl list` and `camdl batch status` both do this live; `batch status` re-plans
the sweep from the batch TOML and reports, per cell, whether `store.lookup` on
its resolved identity is already a `Hit`.

A multi-cell `simulate` additionally writes one `SimEnsemble` leaf holding the
combined wide-format TSV, with a `deps` edge to each cell's `traj.tsv`.

### 9.8 Known identity gaps

Two resolved inputs change stored bytes without changing a `run_id`. Until they
are fixed, a store that has seen both settings holds whichever landed first.

- **`--integrator rk4|rk45` (`Sim`).** The override is applied to the model
  inside the execution path (`util::resolve_run_model` →
  `apply_integrator_override`) but the identity path loads the raw IR
  independently (`main.rs::build_simulate_cas_sink`), so `SimConfig` and the
  model digest both key as if the flag were absent. The two integrators produce
  different trajectories. `--force` does not repair it: the recomputed bytes are
  discarded by `commit_atomic`. The fix is the `--output-every` pattern (§9.3) —
  rematerialize the IR once so both paths load the same model — or a resolved
  `integrator` field on `SimConfig`.
- **`--dates` (`SimEnsemble`).** The flag adds a `date` column to the combined
  TSV that becomes the ensemble artifact, but nothing in the ensemble's four
  levels records it; the per-cell `Sim` leaves are unaffected because the leaf
  trajectory writer never emits a date column. `--dates` belongs in the ensemble
  `config` level.

Related but sound: `event_log.tsv` and `reactive_log.tsv` are identity-free
extra artifacts, so a leaf committed without them cannot gain them later. Call
sites check for the file rather than assume it.

---

## 10. Output File Schemas

Every tabular output is tab-separated. The column _roles_ of the files a run
writes are declared in `run.json.output_schema` (§10.8), read back from the
file's own header, so a consumer never has to reverse-engineer a column name.

### 10.1 Trajectories (`traj.tsv`)

The canonical per-cell trajectory artifact in a `Sim` leaf. One row per output
time. **No comment or version line** — the first line is the header:

```
t	S	I	R	flow_infection	flow_recovery
0	990	10	0	0	0
1	989	9	2	1	2
```

Columns, in this order (`util::TrajColumns::select`):

1. `t` — the output time, rendered with `{}` on an `f64` (`0`, `1`, `0.5`).
2. Integer compartments, in model declaration order, rendered `{}`.
3. Real compartments, in model declaration order, rendered `{:.4}`.
4. `flow_<transition>` for every transition, in model declaration order —
   integer flows `{}`, real flows `{:.4}`. Each value is the flow accumulated
   since the previous output row.

`--no-flows` drops group 4; `--columns A,B,…` restricts groups 2–4 to an
allow-list validated against the model (an unknown name is a hard error listing
the valid ones). Emitted order always follows the model, never the allow-list.
Both knobs are identity (§9.2.4), so a filtered trajectory is its own leaf.

The leaf trajectory never carries a `date` column, regardless of `--dates`.

### 10.2 Combined trajectory views (`ensemble.tsv`, `-o`, `--stdout`)

A multi-cell `simulate` (`--replicates` / `--seeds` / multiple `--scenario` /
`--draws`) interleaves every cell into one wide TSV. The same buffer is the
`ensemble.tsv` artifact of the `SimEnsemble` leaf, the `-o PATH` mirror, and the
`--stdout` stream, so all three are byte-identical.

```
# 0.1.0+<git> (<build date>)
replicate	t	date	S	I	R	V	flow_infection	flow_recovery
1	0	2020-01-01	9990	10	0	0	0	0
```

- Line 1 is a `# {camdl version} ({build date})` comment — present here and
  absent from `traj.tsv`.
- Leading key columns appear only when the corresponding axis has more than one
  level: `replicate` when the run has more than one cell, `scenario` when more
  than one scenario, `draw` when more than one parameter draw.
- `date` appears only under `--dates`, which requires the model to declare an
  `origin`; it is rendered via `ir::caltime::internal_to_date_hires`.
- The data columns are identical to §10.1 and honour the same
  `--no-flows`/`--columns` filter.

A single-cell `simulate` writes no ensemble leaf; the one `Sim` leaf is the
whole result.

### 10.3 Posterior draws (`draws.tsv`)

Written by the Bayesian samplers at the end of a stage, into the stage leaf. IF2
and the optimizer stages write none.

PGAS and PMMH write **all** model parameters — estimated first in `[estimate]`
order, then every remaining model parameter as a constant column — behind two
key columns:

```
chain	draw	beta	gamma	rho	k	N0	I0
0	500	3.41…e-1	1.02…e-1	6.00…e-1	1.00…e1	1.00…e4	1.00…e1
```

- `chain` is **0-based** in every method, matching `PosteriorDraw.chain` in
  `trajectories.tsv` — the join key. (The on-disk `chain_N/` directory is
  1-based; the in-file key is not.)
- Rows are the post-burn-in, thinned set. PGAS writes the set the sim-side
  recorder already retained and re-applies neither burn-in nor thinning; PMMH
  rebuilds the file from each `chain_N/trace.tsv`, dropping rows with
  `step < burn_in`.

A real PGAS `draws.tsv` (2 chains, `burn_in = 10`, `thin = 2`; `beta`/`gamma`
estimated, `rho`/`k`/`I0` fixed):

```
chain	draw	beta	gamma	rho	k	I0
0	10	2.29325244651855442e-1	2.12145731754385697e-1	5.99999999999999978e-1	2.00000000000000000e1	5.00000000000000000e0
0	12	2.29325244651855442e-1	2.12145731754385697e-1	5.99999999999999978e-1	2.00000000000000000e1	5.00000000000000000e0
```

Per-method differences that matter to a consumer:

| method | `draw` column                                                                               | value precision                                                                                                |
| ------ | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| PGAS   | the draw's true 0-based **sweep index**, so it joins `trajectories.tsv` on `(chain, sweep)` | all columns `{:.17e}`                                                                                          |
| PMMH   | a within-chain **row index** (no latent path is saved, so nothing to join to)               | estimated columns copied verbatim from the trace (shortest round-trippable `Display`); fixed columns `{:.17e}` |
| NUTS   | a within-chain row index                                                                    | `{:.17e}`, **estimated parameters only** — no fixed columns                                                    |

Because PGAS and PMMH carry fixed parameters as constant columns, their file is
a complete parameter specification: a posterior-predictive run needs no
companion `--params`. A NUTS `draws.tsv` is not self-contained.

### 10.4 Sampler traces (`chain_N/trace.tsv`, `chain_N/parameter_traces.tsv`)

One file per chain, streamed during sampling (header flushed immediately, rows
batched, so a live tail works). `chain_N` is 1-based.

PGAS, PMMH and NUTS write `chain_N/trace.tsv` via the shared
`fit::trace_writer::TraceWriter`:

```
{index_col}	{loglik_col}	log_posterior	{extra…}	{param…}
```

The likelihood column is _named by the method_ because its meaning differs: PMMH
writes the marginal particle-filter estimate (`log_likelihood`), PGAS the
complete-data conditional value (`log_complete_data_ll`), which is many orders
of magnitude more negative. A shared bare name would invite comparing the two.
`extra` columns are method-specific; `param` columns follow in `[estimate]`
order. A real PGAS-with-NUTS trace header, from a two-stream fit estimating
`beta` and `gamma`:

```
sweep	log_complete_data_ll	log_posterior	trajectory_renewal	renewal_b0	renewal_b1	renewal_b2	renewal_b3	renewal_b4	renewal_b5	renewal_b6	renewal_b7	renewal_b8	renewal_b9	as_opportunity	as_accept	as_proposed	transition_ll	obs_ll	initial_state_ll	obs_ll_cases	obs_ll_confirmations	tree_depth	n_leapfrog	step_size	accept_stat	n_divergent	energy	beta	gamma
```

Two of PGAS's diagnostic blocks need reading before the positions make sense,
and one of them varies in width with the model — so a reader must take the
header as authoritative rather than assume fixed column positions:

- `renewal_b0 … renewal_b9` — always ten columns: `trajectory_renewal` resolved
  in time, one bin per tenth of the substep series. A bin holding no substep
  renders `NA`, not `0.0`: "no substep fell here" and "no substep here was
  renewed" are different diagnoses.
- `obs_ll_<stream>` — the observation term `obs_ll` resolved by declared
  observation stream, one column per stream, each summing over that stream's own
  observation times and (for an indexed stream) its strata. A row is a sweep and
  a sweep evaluates the whole likelihood, so every column carries a number on
  every row even when the streams are on different cadences. They add up to
  `obs_ll` to floating-point round-off — the scalar sums time-major and the
  decomposition stream-major.

The index column is `sweep` for PGAS and `step` for PMMH. Diagnostic columns
render `{:.4}`; parameter columns use the shortest round-trippable `Display` (a
fixed `{:.6}` previously zeroed any parameter below ~5e-7, faking a frozen chain
and corrupting R̂/ESS).

IF2 and the NLopt stages write `parameter_traces.tsv` instead, e.g.

```
iteration	loglik	if2_perturbed_loglik	beta	gamma
```

`--resume` opens the existing file in append mode and writes no second header.

### 10.5 Observation draws

`simulate --obs/--obs-dir/--obs-only/--obs-only-dir` draws synthetic
observations from the model's `observations {}` block. They live in the leaf
under `obs/{obs_hash8}-{obs_seed}/<stream>.tsv`, one file per stream, and are
mirrored to the requested path or directory.

Per-stream file:

```
time	weekly_cases
0	0
7	10
```

The time column is named `time` here, not `t`. Values render as an integer when
the draw is integral and `{:.6}` otherwise. The leaf's stream files never carry
a `date` column — `--dates` reaches only the mirrors.

The single-file wide form (`--obs PATH`) prefixes the same optional
`replicate`/`scenario`/`draw` key columns as §10.2, then `time`, an optional
`date` under `--dates`, then one column per stream. It requires every stream to
share a schedule; a multi-cadence model is a hard error directing the user to
`--obs-dir`.

The `obs/` child is keyed on `(trajectory run_id, obs model hash, obs seed)`, so
observation draws can be iterated without recomputing the trajectory.

**`--emit-every` (gh#656).** A model's `emit_schedule` is the SIMULATE-only
emission cadence — the fit path reads the data file's own time column and never
consults it — so one model can serve a daily and a weekly emission without
editing its source. `--emit-every N` sets every stream, `--emit-every NAME=N`
one stream by its observation-block label (the IR `source`, so one flag covers a
stratified family); the two forms are mutually exclusive, `N` is a plain number
in the model's `time_unit`, and a stream whose schedule is a fixed `at [...]`
list is refused by name rather than silently converted to a cadence.

The override is applied at the CONSUMPTION sites — the `--obs*` writers, the
`obs/` child, an obs-sourced quantity, and `[synthetic]` generation — **not** by
rematerializing the compiled IR the way `--output-every` does. That is
deliberate: `emit_schedule` never enters a likelihood, so moving the model hash
for it would re-key a `fit` against real data over a change that fit cannot see,
orphaning a completed fit. Each path therefore keys what the override actually
determines:

| path                         | what re-keys                                                                                       |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `simulate --obs*` / `obs/`   | the `obs_hash` naming the child, so two cadences are two subtrees under one shared trajectory leaf |
| `fit run` with `[synthetic]` | the generated data's own bytes, which `FitDigest.data` already hashes — correct, the data changed  |
| `fit run` on real data       | nothing; the flag is refused there, since it could only do nothing                                 |

Two cadences share one trajectory leaf because the trajectory bytes do not
depend on the emission cadence. The `obs/` child is a declared boundary in the
leaf's exact-set (keyed by the directory NAME), so a second subtree beside the
first is not an orphan.

### 10.6 Fit sidecars

Fit-wide, at `fits/{fit_stem}-{h8}/` (outside any leaf, so not exact-set
checked):

| file                   | content                                                                                                                                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit.meta.json`        | model path + `model_identity`, fit.toml path + hash, per-stream data hashes, estimated/fixed sets, resolved priors, parameter provenance, and the observation-stream schema |
| `fit.toml.original`    | verbatim copy of the config as run                                                                                                                                          |
| `model.camdl.original` | verbatim copy of the model source                                                                                                                                           |
| `model.ir.json`        | the compiled IR the fit ran against                                                                                                                                         |
| `model.graph.json`     | the model flow graph                                                                                                                                                        |
| `model.render.json`    | the display/LaTeX rendering payload                                                                                                                                         |

Per stage leaf, alongside `run.json`. The set is method-dependent; every file
present is listed in `run.json.artifacts` and subject to the exact-set gate.

Common to both families:

| file               | content                                                                    |
| ------------------ | -------------------------------------------------------------------------- |
| `fit_state.toml`   | θ̂ + stage state; the artifact a downstream stage consumes as a `deps` edge |
| `chain_starts.tsv` | the initial point each chain was given                                     |

An optimizer stage (IF2, NLopt) additionally writes:

| file                           | content                                                                                                                                                                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mle_params.toml`              | flat `name = value` params plus a `[provenance]` block (camdl version, timestamp, tamper `content_hash`, `fit_hash`, backend, dt, model + `model_identity`, per-stream data hashes, seed, stage, chain, log-likelihood, `loglik_sd`, `n_particles`, ESS) |
| `final_params.toml`            | the selected parameter vector                                                                                                                                                                                                                            |
| `chain_N/final_params.toml`    | per-chain endpoint                                                                                                                                                                                                                                       |
| `chain_N/parameter_traces.tsv` | per-chain trace (§10.4)                                                                                                                                                                                                                                  |
| `chain_evaluations.tsv`        | per-chain loglik evaluation summary                                                                                                                                                                                                                      |
| `diagnostics.tsv`              | per-parameter convergence diagnostics                                                                                                                                                                                                                    |

A sampler stage (PGAS, PMMH, NUTS, MH) additionally writes:

| file                        | content                                                                                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `draws.tsv`                 | the thinned posterior cloud (§10.3)                                                                                                                |
| `chain_N/trace.tsv`         | per-chain trace (§10.4)                                                                                                                            |
| `chain_N/resume_state.bin`  | bincode resume state, guarded by its own config hash (§9.4)                                                                                        |
| `chain_N/trajectories.tsv`  | PGAS only — the smoothed latent paths, tidy/long, keyed `chain draw time [date]`, with a `# camdl-trajectories v1` header line                     |
| `chain_N/trajectories.json` | the matching manifest (`format`, `version`, `method`, `granularity`, `n_chains`, `n_draws`, `columns`, `model_hash`, `conditioned`, `calendar`, …) |
| `<algorithm>_summary.json`  | `pgas_summary.json`, `pmmh_summary.json`, `mh_summary.json`, `nuts_summary.json` — one file per algorithm, deliberately never shared               |
| `diagnostics.json`          | R̂ / ESS / divergence diagnostics                                                                                                                   |
| `progress.json`             | sampler progress, written live                                                                                                                     |

The `mle_params.toml` `content_hash` is a _tamper_ hash — SHA-256 over
`{name}={value:.12}\0` pairs, truncated to 8 hex — so a hand-edited parameter
value is detectable. It is not a content hash of the file.

### 10.7 Generated quantities

`simulate --quantities-out DIR` writes `DIR/quantities/<name>.tsv` plus a
`DIR/quantities.json` manifest (`schema: "camdl.quantities/v1"`). A single
fixed-parameter run emits a bare `value` per leaf (point mode); a
`--draws`/`--replicates`/`--seeds` run emits banded quantiles.

Quantities are a **regenerated sidecar**: they never enter a CAS leaf and never
enter run identity (`quantities` and `contrasts` are the model fields excluded
from `Model::hash_into`, §9.2.3). This is sound only because `camdl simulate`
recomputes every cell even on a cache hit (§9.3) — the quantities are always
evaluated from the run just performed, never read back from a stored leaf.

Without the flag, a model that declares quantities prints a note and skips them
rather than erroring.

### 10.8 Declared column roles (`run.json.output_schema`)

Rather than reconstruct a writer's column order — which would drift from the
writer — camdl reads each written file's **actual header** and classifies every
column, so the declaration cannot disagree with the file it describes
(`cli/src/output_schema.rs`). The entry is keyed by leaf-relative path, with
`{n}` standing in for the chain index.

`role` (the table's default view) is one of `trajectory`, `observation`,
`posterior_cloud`, `trace`, `predictive`, `landscape`.

Each column's `role` is one of `time`, `iteration`, `chain`, `replicate`,
`scenario`, `dimension`, `state`, `flow`, `incidence`, `param_estimated`,
`param_fixed`, `observable`, `quantile`, `diagnostic`. Classification is by
reserved name (`chain`, `replicate`, `scenario`; `t`/`time`/`date` → `time`;
`sweep`/`step`/`draw`/`iteration`/`point_id` → `iteration`), then prefix
(`flow_`, `inc_`), then membership in the estimated / all-parameter sets, then a
per-table default (`state` for trajectories, `diagnostic` for fit tables).

`time` and `iteration` are deliberately distinct: a trajectory's x-axis is
physical time and a trace's is a sampler index, and conflating them is a real
rendering bug.

The rendering rule is then mechanical: the x-axis is the `time` or `iteration`
column; group by `chain` / `replicate` / `scenario`; facet by `dimension`;
series are `state` / `param_estimated` / `observable`; ribbons are `quantile`;
overlays are `diagnostic`. Because the index column is spelled differently by
method (`sweep`, `step`, `draw`, `iteration`), a consumer reads the role and
never the name.

Two producers emit a schema today. A `sim` leaf declares `traj.tsv` (table role
`trajectory`, unrecognized columns defaulting to `state`). A completed fit stage
declares `draws.tsv` (role `posterior_cloud`) plus one entry per per-chain trace
filename — `trace.tsv` for PGAS/PMMH/MH/NUTS, `parameter_traces.tsv` for
IF2/nlopt — read from the first chain directory with a readable file, so an
early crashed chain does not drop the entry. Both are best-effort: an unreadable
file is omitted rather than failing the run.

`pfilter`, `survey`, `profile`, `sim_ensemble`, and the `fit predict` outputs
declare no schema. Of the vocabulary above, the column roles `dimension`,
`observable`, and `quantile` and the table roles `observation`, `predictive`,
and `landscape` are defined but unreachable — no producer emits them.

The schema is recorded, never hashed, so attaching it cannot re-key a run. It is
omitted from `run.json` when empty.

> There is no `camdl summarize` and no automatic per-run summary table (`peak_X`
> / `tpeak_X` / `final_X` / `integral_X`). The `camdl experiment
> summarize`
> subcommand that produced those was removed. Equivalent reductions are
> expressed in the model's `quantities {}` block (§10.7).

## 11. Predictive Workflows

Every workflow in this section is a `camdl simulate` run over a **cloud of
parameter vectors** (`--draws`) rather than a single point. The four sources are
`prior`, `posterior`, `uniform`, and a path to a parameter TSV. A run with
`--draws`, `--replicates`, or `--seeds` is an _ensemble_: one content-addressed
store leaf per cell, plus an ensemble node that names them.

### 11.1 Prior Predictive Check

_"Does my model, under priors, generate data that looks plausible?"_

Two forms. With no fit config, priors come from the model's `~` declarations:

```bash
camdl simulate models/sir.camdl --draws prior -n 500 \
    --replicates 5 --obs-dir prior_pred/
```

With `--fit`, the fit config's `[estimate.<p>.prior]` blocks take precedence and
the model's `~` priors are the fallback (§12):

```bash
camdl simulate models/sir.camdl --draws prior --fit fits/02_fix_beta.toml \
    -n 500 --replicates 5 --obs-dir prior_pred/
```

`-n` is **required** for `--draws prior` and `--draws uniform`; omitting it is a
hard error (`error: --draws prior requires -n N`).

**Model-only form: every parameter needs a prior _or_ a value.** A parameter
with neither is a hard error naming all of them at once:

```
$ camdl simulate seir_observations.camdl --draws prior -n 3
error: parameters 'beta', 'sigma', 'gamma', 'rho', 'k', 'p_detect', 'N0', 'I0' no prior and no default value.
  Fix options: add `~ prior(...)` to the model, supply `--scenario NAME` if a scenario pins these values,
  supply `--fit FIT.toml`, or use `--draws uniform` for space-filling exploration.
```

A `--scenario` satisfies the requirement for the parameters it pins, so
`--draws prior --scenario baseline` samples the parameters that declare a prior
and holds the scenario's `set = { … }` values fixed. When several `--scenario`
names are given on the model-only prior path they are **layered** in order
(later wins) rather than run as separate cells — this differs from every other
`simulate` path, where `--scenario a,b` is two cells.

**`--fit` form: every estimated parameter needs a proper prior.** Flat is
refused, because there is no finite distribution to sample:

```
$ camdl simulate seir_observations.camdl --draws prior --fit if2_a.toml -n 3
error: --draws prior requires a proper (non-flat) prior on every estimated parameter.
  Missing or flat priors: beta
  To fix, either:
    (i)  add `prior = { <dist> = { ... } }` to `[estimate.beta]` in your fit.toml (e.g. `prior = { log_normal = { mu = 0, sigma = 1 } }`), OR
    (ii) add a `~ <dist>(...)` declaration to parameter `beta` in your .camdl model file.
```

An explicit `prior = { flat = {} }` is rejected here too, with an added note
explaining why (improper uniform, infinite support).

**Recording the draws.** `--draws-out <PATH>` writes the sampled vectors as a
TSV with one row per draw and one column per parameter — including the
parameters held fixed, so the file round-trips through `--draws <PATH>`:

```
$ camdl simulate seir_observations.camdl --draws uniform -n 3 --draws-out u.tsv
$ head -2 u.tsv
I0	N0	beta	gamma	k	p_detect	rho	sigma
2.29285758020031290e3	7.34369783379929606e5	4.42244622597675330e-1	…
```

Prior draws are rejection-sampled against each parameter's declared bounds; the
run reports per-parameter rejection counts when truncation was active.

### 11.2 Posterior Predictive Check

_"Does my fitted model generate data that looks like the real data?"_

Prefer `--draws posterior --fit <fit results dir>`, which resolves the fit's
canonical post-warm-up cloud for you rather than making you find the file:

```bash
camdl simulate models/sir.camdl \
    --draws posterior --fit results/fits/fit_pgas-4dadedae \
    --replicates 10 --obs-dir ppc/
```

It prints what it resolved:

```
draws: posterior — 2 draws from pgas stage 'post' (…/01-post-a0b1da4f/seed_1-06cbd6b3/draws.tsv)
```

Resolution is **by artifact, not by method name**: a stage has a posterior iff
it wrote `draws.tsv`. An optimizer-only fit (IF2, `nl-sbplx`, `nl-bobyqa`)
resolves to an error rather than dressing a single point up as a distribution.
The canonical file is the stage leaf's `draws.tsv` — post-warm-up and thinned —
**not** `trace.tsv`, which carries warm-up rows for live observability. There is
no `<fit-dir>/posterior/` directory; the path is
`<fit-dir>/<NN>-<stage>-<h8>/seed_<N>-<h8>/draws.tsv`.

A raw TSV path still works. If the file carries only the estimated columns (a
posterior trace tail), pass `--fit` alongside it and the fit's `[fixed]` block
backfills the missing parameters — never overwriting a column the file provides:

```bash
camdl simulate models/sir.camdl --draws draws.tsv --fit fit.toml \
    --replicates 10 --obs ppc.tsv
```

**The purpose-built alternative.** `camdl fit predict` writes the
predicted-vs-observed artifact directly from a completed fit, banded over the
posterior, with no re-plumbing of draws:

```bash
camdl fit predict results/fits/fit_pgas-4dadedae
```

It emits, under the run directory:

```
predictive/<stream>.tsv   time | <dims…> | horizon | treatment | rhat_max | q05..q95
observed/<stream>.tsv     time | <dims…> | value
```

with two horizons where the backend supports them — `free_forward` (one forward
replay per draw) and `one_step` (the filter's one-step-ahead predictive; chain-
binomial only, a hard error on an ODE fit). The posterior cloud is subsampled to
`--n-draws` (default 200) by a strided pick across the whole cloud.

### 11.3 Scenario Prediction Under Posterior Uncertainty

_"What would happen under an SIA, given what we learned from the data?"_

```bash
camdl simulate models/sir.camdl \
    --draws posterior --fit results/fits/<run>/ \
    --scenario baseline,with_sia \
    --replicates 10 --obs-dir obs/
```

`--scenario a,b` is a **partition of the cell grid**: each scenario is its own
set of runs, not a layering. For each (draw, seed) pair both scenarios are
simulated with the same seed, giving paired comparisons that propagate posterior
uncertainty.

**The coupling is paired-seed CRN, not event-keyed RNG.** The runtime uses a
stateful ChaCha8 PRNG, so two arms agree only while they consume draws in the
same order. `enable`/`disable` scenarios are byte-identical up to the first
state divergence; `set`/`scale` scenarios that perturb propensities from `t=0`
are correlated but never identical. Any structural change that reorders draws
also breaks the coupling.

**Do not hand-difference the two arms for a headline number.** The model-level
`contrasts { }` block does this correctly and is auto-emitted by
`camdl fit predict`:

```camdl
quantities {
  total = final(D)               # total deaths over the run
}

contrasts {
  averted = no_sia.quantities.total - with_sia.quantities.total
}
```

The fork point is **derived, not declared**: the reducer diffs the two arms'
live intervention sets, finds the toggled intervention, and forks both arms at
the last saved trajectory snapshot strictly before its fire time — so "fork at
or after the intervention" is unrepresentable. Both arms share the smoothed
latent state at the fork and the per-draw seed, then desync by design after it.
Results are banded over the forkable draws into `contrasts/<name>.tsv`. A
contrast with no toggled intervention (a parameter-only scenario) or one toggled
by a parametric or reactive fire time is skipped with a note, never silently.

**A chain selection reaches the contrast.** `fit predict --exclude-chains`
narrows the forkable set to the retained chains before the fork, so a contrast
bands over exactly the cloud `predictive.json`'s `chain_selection` block
describes — the contrast's `n_used` and the free-forward rows' `n_draws` are one
number. The retained-chain scope is printed alongside the count. Excluding
chains narrows an already-partial set (only path-saved draws are forkable); if
the intersection is empty the run is refused by name, never emitted as an empty
band.

**`--quantities-out` refuses a multi-scenario run.** `quantities { }` bands are
taken over draws × replicates × seeds; scenario is a partition of that grid, not
part of the band. Pooling them would produce one ribbon describing neither arm,
so the run errors and prints the per-scenario invocations to use instead.

### 11.4 Uniform Exploration

_"What does the model do across parameter space?"_

```bash
camdl simulate models/sir.camdl --draws uniform -n 500 --replicates 1
```

Samples uniformly from each parameter's declared bounds. No Bayesian pretension
— this is space-filling exploration for model debugging. For a likelihood-aware
version of the same question (is the model identifiable from this data at all?)
use `camdl survey`, which runs a Latin-hypercube over `[estimate]` bounds and
scores each point.

### 11.5 Replicate Fits at Known Truth

The orchestration that generates synthetic datasets and fits each one **exists**
— it is the `[synthetic]` block in a fit config, not a separate subcommand:

```toml
[synthetic]
true_params = "truth.toml" # ground truth used to generate the data
sim_seeds = "1:20" # one dataset per seed
backend = "chain_binomial" # generation backend (fits declare their own)
scenario = "baseline" # optional: scenario for GENERATION only
```

`camdl fit run` then generates one wide-format TSV per dataset into
`<fit_dir>/synthetic/data/`, and runs the declared stages once per dataset. Each
grid cell is its own content-addressed fit, readable with `camdl list` / `show`
/ `cat`.

Two honest limitations:

- This is **calibration at a fixed truth**, not simulation-based calibration.
  Classical SBC (Talts et al. 2018) draws θ from the prior _per dataset_ and
  checks rank uniformity; `[synthetic]` holds `true_params` fixed across every
  dataset and asks whether the estimator recovers that one point.
- The cross-cell roll-up — the parameter-recovery coverage and bias table — is
  **not built**. The run prints
  `note: grid summary / coverage are derived views — rebuilt by the reindex in M4
  (gh#150 / gh#154)`.
  Until then the per-cell fits are there and the aggregation is the reader's to
  do (`camdl fit table <root> --format csv` is the starting point).

---

## 12. Priors: Beliefs Belong With Parameters

Priors are declared in the model file with `~` syntax on the parameter:

```camdl
parameters {
    beta  : rate in [0.01, 2.0] ~ log_normal(mu = -1.0, sigma = 0.5)
    gamma : rate in [0.05, 1.0] ~ half_normal(sigma = 0.3)
    rho   : probability in [0.001, 1.0] ~ beta(alpha = 2.0, beta = 5.0)
}
```

**Why in the model file?** Priors are beliefs about parameters — they answer
"what do I think about beta before seeing data?" That belief belongs with the
parameter declaration, where anyone reading the model can see it. This design
follows Stan, PyMC, and Turing.jl.

Camdl exists to support decisions about people's lives. Getting uncertainty
wrong means making confident-looking recommendations on shaky foundations. Prior
predictive checks are the first line of defence: "do my stated beliefs produce
data that looks plausible before I've seen any real data?" Making priors
discoverable and declarative in the model file is part of doing uncertainty
right.

### 12.1 Precedence

Two chains, resolved by the same code (`rust/crates/cli/src/fit/runner.rs:2744`
`resolve_prior`) but with a **different tier-3**. The difference is deliberate
and load-bearing.

`camdl fit run`, Bayesian stages (`pgas`, `pmmh`, `mh`, `nuts`):

```
1. fit.toml [estimate.<p>.prior] = { <dist> = { ... } }   → source "fit.toml"
2. fit.toml [estimate.<p>.prior] = { uniform = {} }       → uniform over bounds
3. model IR parameter.prior (from `~ <dist>(...)`)        → source "model"
4. fit.toml [estimate.<p>.prior] = { flat = {} }          → source "flat (explicit)"
   ── otherwise ──
   HARD ERROR before the fit starts.
```

There is **no implicit flat fallback on the fit path.** A `fit run` chain is
treated as the canonical posterior downstream (`fit_summary.json`,
`fit predict`, `compare`), so silently targeting the unconditioned likelihood is
not allowed. The refusal names every parameter and all three remedies:

```
$ camdl fit run fit_noprior.toml --seed 1
error: stage 'post' (method=pgas) has parameters with no resolved prior:

  beta        no prior in fit toml, no `~` in model file

To proceed, do one of:

  (i)   Declare `prior = { <dist> = { ... } }` in the fit toml's
        [estimate.<param>] for each listed parameter.
  (ii)  Declare a `~ <dist>(...)` prior in the model file for
        each listed parameter.
  (iii) Opt into flat priors explicitly via
        `prior = { flat = {} }` in the fit toml — only do this if you
        intentionally want the chain to target the unconditioned
        likelihood (scaled-likelihood posterior).
```

`camdl profile --algorithm pmmh`, per-cell PMMH — same first three tiers, then:

```
4. Prior::Flat  → source "flat (default)", with a WARNING naming each parameter
```

Profile warns rather than errors because a per-cell MLE-as-MAP is recoverable by
spot-checking the per-cell values. Suppress the warning with
`--suppress-warnings` (or
`[diagnostics] suppress = ["profile_flat_prior_fallback"]` in a supplied fit
toml) — the choice is recorded in `run.json` either way.

Per-parameter provenance is written to `run.json` under `resolved_priors`, with
sources `fit_toml`, `model_ir`, `flat_explicit`, `flat_fallback`.

### 12.2 When priors are used at all

| Stage algorithm              | Prior used?                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| `pgas`, `pmmh`, `mh`, `nuts` | Yes — the prior density enters the acceptance ratio; a missing prior is a hard error |
| `if2`                        | No — iterated filtering is maximum likelihood; the code never reads a prior          |
| `nl-sbplx`, `nl-bobyqa`      | No — deterministic MLE                                                               |
| `pfilter`                    | No — likelihood evaluation only                                                      |

`--draws prior` needs a proper prior on every _sampled_ parameter; parameters
with a concrete value (model default, `--scenario`, or fit `[fixed]`) are held
constant instead (§11.1).

### 12.3 Supported distributions

Nine families, all accepted in the DSL `~` position with keyword arguments only:

| Distribution       | DSL syntax                             |
| ------------------ | -------------------------------------- |
| `uniform`          | `~ uniform(lower = L, upper = U)`      |
| `normal`           | `~ normal(mu = M, sigma = S)`          |
| `log_normal`       | `~ log_normal(mu = M, sigma = S)`      |
| `half_normal`      | `~ half_normal(sigma = S)`             |
| `beta`             | `~ beta(alpha = A, beta = B)`          |
| `gamma`            | `~ gamma(shape = K, rate = R)`         |
| `exponential`      | `~ exponential(rate = R)`              |
| `log_uniform`      | `~ log_uniform(lower = L, upper = U)`  |
| `truncated_normal` | `~ truncated_normal(mean = M, sd = S)` |

A hierarchical (partially pooled) prior appends a dimension:
`~ log_normal(mu = mu_h, sigma = sigma_h) | age`, whose arguments are
expressions over hyperparameters re-evaluated at each MCMC step.

**The DSL and the fit-toml spellings of `normal` differ, and the compiler will
tell you.** In the DSL it is `mu`/`sigma`; in fit-toml it is `mean`/`sd`:

```
$ camdlc check m.camdl     # with `~ normal(mean = 0.3, sd = 0.1)`
error[E231]: parameter 'beta': prior 'normal' missing required argument 'mu'
```

```toml
# fit.toml
[estimate.beta]
prior = { normal = { mean = 0.3, sd = 0.1 } }
```

Two fit-toml-only forms have no DSL spelling:

- `prior = { flat = {} }` — the explicit improper-uniform opt-in (§12.1).
- `prior = { uniform = {} }` — uniform over the parameter's bounds, taken from
  `[estimate.<p>].bounds` if present, else the model's `in [lo, hi]`. Declaring
  neither is an error naming both places you could fix it.

Parameterization conventions (log-scale `log_normal`, rate-parameterized
`gamma`, `truncated_normal` truncated at the declared `in [lo, hi]`) are in the
language spec's priors section, which is the normative source for the surface.

### 12.4 Prior–transform compatibility

Each estimated parameter carries a transform (derived from its `param_kind`, or
overridden by `transform` in `[estimate.<p>]`). A prior whose support does not
match the transform is refused before the fit starts, because the mismatch would
silently produce a _different_ prior than the one written — `log_normal` on an
identity transform collapses to a normal on the natural scale.

| Prior family                          | Required transform |
| ------------------------------------- | ------------------ |
| `log_normal`                          | `Log`              |
| `beta`                                | `Logit`            |
| `half_normal`, `gamma`, `exponential` | `Log`              |
| `uniform`, `normal`, `flat`           | any                |

The error names the parameter, both sides of the mismatch, and the resolved
prior's source.

---

## 13. Utility Commands

### 13.1 `camdl fit table`

Walks a results tree and renders one row per fit — terminal-stage method,
convergence verdict, best log-likelihood and its type, age. The `<ROOT>`
argument is **required**.

```
$ camdl fit table results/fits
fit_id     label                  stem           method   stages converged     best_ll ll_type          age
--------------------------------------------------------------------------------------------------------------
4dadedae   <unlabelled>           fit_pgas       pgas     post   yes                 — complete_data    15s
```

Read-only by default: every cell is recovered from the on-disk `run.json` and
per-stage outputs. **One flag breaks that rule** — `--quantity <NAME>` may
_derive_ a value on demand, running `fit predict --horizon free_forward` for any
fit that has not been predicted yet, and writing that fit's `quantities/`
outputs. Optimizer fits have no posterior cloud, so their cell renders `—`.

Filters: `--converged`, `--gate-failed`, `--with-stage <STAGE>`,
`--with-method if2|pgas|pmmh`, `--model <HASH_PREFIX>`, `--hash <HASH_PREFIX>`,
`--since-seconds <N>`, `--label-pattern <GLOB>`. `--baseline <HASH_PREFIX>`
picks the fit that the Δ columns are measured against (default: lowest hash in
the surviving cohort). `--exclude-chains <IDS>` applies only to derived
`--quantity` cells and always warns, because post-hoc chain exclusion biases the
posterior toward the retained mode.

`--format text|json|md|csv`. The JSON document is schema-pinned; each row
carries considerably more than the text view:

```
schema, fit_id, fit_hash, label, stem, model_identity, stages, method,
config_diff_from_baseline, converged, gate_verdict, best_loglik, loglik_type,
max_chain_agreement, max_rhat, acceptance_rate, ess_at_mle, ess_posterior,
ess_per_iter, ess_per_sec, params, delta_ll_vs_best, age_seconds, created_at,
stale, stale_reason
```

A fit directory with no completed stage leaf is reported as a warning line and
skipped, not silently dropped.

To enumerate runs of any kind without the per-stage projection, use the generic
run browser: `camdl list --kind fit`. For one fit's full interpretation, use
`camdl fit summary <handle>`.

### 13.2 `camdl fit diff`

Compares two fit configs. Both must parse as complete v2 configs (`[estimate]`
is required), so this is a config-to-config diff, not a text diff.

```
$ camdl fit diff fit_a.toml fit_b.toml
diff: fit_a.toml → fit_b.toml

  beta: [estimate] → [fixed] = 0.31
  sigma: [fixed] = 0.2 → [estimate]

Stages:
  stage 'eval': (new) pfilter
  stage 'post': chains 1→4
```

Parameter-side coverage is complete: estimate↔fixed moves, changed fixed values,
changed bounds (including the omit↔explicit transition, which is meaningful
because omitting means "fall back to the model file"), and changed priors.

**Stage-side coverage is not.** Only `algorithm` and `chains` are named; every
other stage setting collapses to the string `settings changed`:

```
$ camdl fit diff if2_a.toml if2_c.toml     # particles 1000→2000, cooling 0.70→0.95
diff: if2_a.toml → if2_c.toml

  (no parameter changes)

Stages:
  stage 'mle': settings changed
```

and when `chains` also changed, the particles/cooling deltas are dropped from
the output entirely rather than appended. The typed per-key stage diff exists
(`ConfigDiff` in `rust/crates/cli/src/fit/config_diff.rs`, covering particles,
sweeps, iterations, cooling, burn_in, thin, tolerances, gate thresholds) and is
what `fit table --format json` reports under `config_diff_from_baseline` — use
that when you need the full delta.

### 13.3 `camdl fit new`

Copies a fit config to a new path and injects a `[provenance]` block. It does
**not** rewrite stages, wire up cross-stage chaining, or fill in `reason`.

```
$ camdl fit new --from fit_pgas.toml fit_v2.toml
  [provenance] derived_from = "fit_pgas.toml"
  hint: set starts_from on your first stage to the last stage leaf under …/results/fits/fit_pgas-4dadedae
        (run `camdl list` to find the exact stage-leaf path)
created fit_v2.toml
```

The written file is the source verbatim with:

```toml
[provenance]
derived_from = "fit_pgas.toml"
reason = ""
```

inserted before the first table. Refuses to overwrite an existing destination.
If the source already has a `[provenance]` block it says so and leaves it alone.

**The hint names a removed key.** Cross-stage chaining is spelled `init_mle`,
not `starts_from`; a config with `starts_from` is rejected at load with a
migration error. Write:

```toml
[stages.refine]
init = "from_mle"
init_mle = "results/fits/<stem>-<h8>/01-scout-<h8>/seed_1-<h8>"
```

or pass it per-run as
`camdl fit run … --stage refine --init from_mle --mle <path>`.

### 13.4 Reading results back

There is no `camdl summarize`. Four commands cover the ground:

| Command                      | Answers                                                                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `camdl list [ROOT]`          | What runs exist? Filter by `--kind sim\|fit\|profile\|pfilter\|survey\|ensemble`, `--model`, `--scenario`, `--since 1h`, `--parent <HASH>`. `--format json`. |
| `camdl show <TARGET>`        | Full metadata for one run, by short hash prefix or path.                                                                                                     |
| `camdl cat <TARGET>`         | The run's trajectory TSV; `--stream <NAME>` selects an observation stream or a named artifact (`event_log.tsv`, `reactive_log.tsv`).                         |
| `camdl label <HASH> <LABEL>` | Name a run so you can find it again. 1–64 chars matching `^[a-zA-Z0-9 ,._-]+$`; refuses a still-running fit and an ambiguous prefix.                         |

Per-scenario summary statistics over a trajectory tree (peak, time-of-peak,
final value, integral) are **not** a camdl command. Declare them in the model as
`quantities { }` and emit with `camdl simulate --quantities-out <dir>`, which
writes `<dir>/quantities/<name>.tsv` plus `<dir>/quantities.json` — a bare
`value` per leaf for a fixed-parameter run, banded quantiles for a
`--draws`/`--replicates`/`--seeds` run. Quantities are a regenerated sidecar and
never part of run identity. A model that declares quantities but is run without
the flag prints a note and skips them.

### 13.5 `camdl mre`

Packages the full input closure of a fit or a forward simulation — model,
`read()` tables, data, params — into a shareable `.tar.gz`:

```bash
camdl mre fit fit.toml                        # → fit.mre.tar.gz
camdl mre fit fit.toml --no-data -b bug.tar.gz  # structure only, no data values
camdl mre simulate sir.camdl --params p.toml --seed 42
```

`-b/--bundle` names the output (`-o` keeps its trajectory-mirror meaning on
`mre simulate`). `--no-data` emits column schema, row counts and time range with
no values, for sensitive data.

### 13.6 `camdl docs`

Guides embedded in the binary — offline, version-matched to the binary that
prints them:

```bash
camdl docs                    # list topics
camdl docs inference          # print one
camdl docs --search rhat      # find where a term is discussed
camdl docs --json             # machine-readable index (for agents)
```

Topics: `agents`, `getting-started`, `language`, `language-changes`,
`changelog`, `inference`, `diagnosing-fits`, `workflow`, `fit-toml`, `concepts`,
`features`, `backends`, `data`, `dates`, `debugging`, `mre`.

---

## 14. Observation Semantics

An `observations { }` stream projects simulator state into the scalar
`projected` value that the likelihood scores. This section states, for each
projection kind, _what is read_, _when_, and _what resets afterwards_ — the
three facts that decide what a likelihood term actually means.

### 14.1 Projection kinds

There are five IR projection variants (`rust/crates/ir/src/observation.rs:11`),
classified into two temporal kinds (`observation.rs:47` `temporal_kind`). The
classification is **derived from the variant, never stored** — an
independently-stored kind could only ever disagree.

| DSL surface                                                                                             | IR variant            | Temporal kind |
| ------------------------------------------------------------------------------------------------------- | --------------------- | ------------- |
| `incidence(tr)` on an unstratified/one-cell transition, or `incidence(tr[a])`                           | `cumulative_flow`     | Interval      |
| `sum(a in dim, incidence(tr[a]))`                                                                       | `cumulative_flow_sum` | Interval      |
| `prevalence(X[cell])`, or a bare `prevalence(X)` on an unstratified compartment                         | `current_pop`         | Instant       |
| bare `prevalence(X)` on a **stratified** compartment                                                    | `current_pop_sum`     | Instant       |
| any expression over state — `X1 + X2`, `I / (S+I+R)`, a `let` reference, several `prevalence` arguments | `derived_expr`        | Instant       |

Verified against the compiler on a two-stratum (`age = [child, adult]`) SIR:

```
projected = prevalence(I)                          → current_pop_sum ['I_child', 'I_adult']
projected = prevalence(I[child])                   → current_pop     I_child
projected = sum(a in age, incidence(infection[a])) → cumulative_flow_sum ['infection_child', 'infection_adult']
projected = I[child] + I[adult]                    → derived_expr
```

- **Interval (incidence)** — the sum of per-transition flow counters accumulated
  over the reporting window, read at the observation and **reset afterwards**
  (`StreamProjection::resets_after_observation`,
  `rust/crates/sim/src/inference/multi_stream_obs.rs:210`). The window is
  right-closed, left-open: `(previous obs, this obs]`, with the first window
  starting at `t_start` (or at `condition_from` when a conditioning window is
  set, which resets the accumulator at the boundary so the first scored bin is
  `(condition_from, first_obs]`).
- **Instant (prevalence / derived)** — a function of the state vector read at
  the observation instant. No accumulation, **no reset**; each observation is
  independent of the previous one.

Reset is per-stream and per-cadence: with several streams on different
schedules, only the Interval streams scheduled at the current union index are
zeroed. A **hole** in the data (an `NA` cell) contributes no likelihood term but
still resets, because the grid time is present.

**A bare `incidence()` over a stratified family on an un-indexed stream is a
hard error, not a silent sum** (E280):

```
$ camdlc check strat_probe.camdl        # projected = incidence(infection)
error[E280]: observation 'cases' is un-indexed, but `incidence(infection)` would silently
sum all 2 strata of 'infection' and apply reporting uniformly
```

Pooling across strata is a modelling decision, so the compiler makes you state
it — either `projected = sum(a in age, incidence(infection[a]))` with one
reporting rate in the likelihood, or an indexed stream `cases[a in age] { … }`
with `rho[a]`. Bare `prevalence(X)` over a stratified compartment is **not**
gated the same way: it sums silently, matching how a bare stratified name
expands to `PopSum` in a rate expression.

**Several `prevalence` arguments desugar to their sum** and lower as
`derived_expr` — `prevalence(Y1 + Y2)`, `prevalence(X1, X2)`, and the compiler-
generated forms for a `via erlang` (an `ESum` over the stage axis) or
`via hyper_erlang` (an Add-chain over branch-stage cells) compartment all take
this path.

`incidence` does **not** follow that convention: it projects exactly one flow,
several arguments is `E203`, and none is `E250` (gh#669). A compartment
population is an expression leaf, so `prevalence` can sum its arguments as an
ordinary expression; a flow has no expression leaf naming it, so there is
nothing for `incidence` to sum into.

`current_pop` / `current_pop_sum` resolve to **integer** compartments
(`multi_stream_obs.rs:253`). A prevalence projection naming a real-valued
compartment is refused with a message saying so; use a `derived_expr` (which
reads both state kinds) instead.

Only `derived_expr` carries a `projection_state_grad`
(∂projection/∂compartment). The three selection variants are linear projections
and need none; the gradient-based samplers refuse a `derived_expr` whose
projection gradient is missing rather than scoring a silent-zero chain-rule
factor.

### 14.2 Snapshot timing

"State at `t`" means three different things depending on which path is running.
Getting this wrong changes what your data means, so it is stated per-path rather
than per-backend.

**Inference (particle filter, IF2, PGAS, PMMH) — exact landing at `t`.** The
filter builds an `Exact`-policy timeline
(`rust/crates/sim/src/inference/particle_filter.rs:198`) whose substep is
`dt.min(next_boundary − t)`, so the walk lands **exactly on the observation
time** even when it is off the `dt` grid. There is no snapping to a step
boundary. Scoring happens after the window walk completes, at the landed state.
A scheduled effect whose fire time is off the `dt` grid is refused loudly at
timeline construction rather than being silently re-tiled.

**ODE inference — the recorded output snapshot must land on the obs time.** The
deterministic likelihood (`rust/crates/sim/src/inference/ode_loglik.rs:93`)
matches each observation to a trajectory snapshot within `1e-9`. It is **not**
dense-output interpolation. If the model's output schedule does not contain an
observation time, the fit errors:

```
ODE trajectory has no snapshot at obs time <t> (snap.t = <s> overshot it). The
model's [output] schedule must include every observation time; declare an
explicit `output { at = [...] }` block aligned to the data, or ensure the
regular schedule's step divides obs intervals.
```

The value path reads `snap.int_state.counts`, which the ODE backend produced by
rounding (`x.max(0.0).round()`, `rust/crates/sim/src/ode.rs:762`). The gradient
path (`nuts`) reads the **unrounded** continuous state instead
(`multi_stream_obs.rs:1424`). At small counts the two therefore score slightly
different likelihood surfaces for the same model and data; at large `N` the
difference is sub-nat.

**Forward `simulate --obs` — the recorded trajectory output grid.** Synthetic
observations are projected from the trajectory snapshots, not from the
integrator: `snap_at` (`rust/crates/cli/src/main.rs:2395`) returns the **last
snapshot at or before** the observation time, and incidence is the difference of
cumulative flow between consecutive such snapshots.

**Consequence: the observation cadence is silently clipped to the trajectory
output cadence.** With a weekly output grid and a daily `emit_schedule`:

```camdl
output { trajectories { every = 7 'days } }
observations {
  prev { projected = prevalence(I)          emit_schedule = every 1 'days  … }
  inc  { projected = incidence(infection)   emit_schedule = every 1 'days  … }
}
```

```
$ head -9 obs/inc.tsv        $ head -9 obs/prev.tsv
time	inc                     time	prev
0	0                       0	12
1	0                       1	11
2	0                       2	10
3	0                       3	7
4	0                       4	10
5	0                       5	9
6	0                       6	11
7	77                      7	69
```

Incidence puts the whole week's flow on day 7 and zero elsewhere; prevalence is
a step function that changes only at output times. Nothing warns.
`--output-every
N` has the same effect, so a flag that reads as presentational
silently changes the emitted data. **Set the trajectory output cadence at least
as fine as the finest `emit_schedule`** before generating synthetic data from a
model.

### 14.3 Interaction with scheduled interventions

If a scheduled intervention fires at the same time as an observation, the
projection reads the **post-intervention** state. The data was generated in a
world where the intervention had already fired; scoring against the
pre-intervention state would deterministically bias the posterior against any
scenario that correctly represents it.

The ordering is fixed in one shared dispatch seam (`Schedule::arrive`,
`rust/crates/sim/src/schedule.rs:536`), not by each backend: **effects before
output**, all coincident effects drained as one batch, then all coincident
outputs. On the inference path the equivalent guarantee comes from the substep
walk: a substep that lands on an effect boundary fires that boundary's batch
inside `process.step` (`particle_filter.rs:315`), and scoring happens after the
window walk. Forward chain-binomial fires effects inside `step_one` for the
substep ending at `t + dt` and records the output afterwards
(`rust/crates/sim/src/chain_binomial.rs:424`).

There is no live `apply_interventions_at` function; it survives only as a test
adapter. Effects are decided by the caller and applied through the `effects`
seam — `round(t/dt)`-keyed on the forward chain-binomial "snap" path,
cursor-keyed off the timeline's effect boundaries on the exact paths.
(Historical: an earlier chain-binomial loop fired interventions twice per
scheduled time; see
`docs/dev/incidents/2026-04-17-chain-binomial-double-fire.md`.)

### 14.4 Likelihood families

Eight families (`rust/crates/ir/src/observation.rs:146`): `poisson`,
`neg_binomial`, `normal`, `binomial`, `beta_binomial`, `beta`, `bernoulli`,
`zero_inflated_neg_binomial`.

Four facts about how they score that are not obvious from the names:

- **`normal(...)` is a discretized _count_ likelihood** (pomp / He et al.), not
  a continuous Normal PDF. It rounds the observation to the nearest non-negative
  integer and integrates the density over `[y − 0.5, y + 0.5]`. A clearly
  fractional observation logs a one-time warning
  (`rust/crates/sim/src/inference/obs_model.rs:93`). For a continuous outcome
  this is the wrong family.
- **Every count family rounds the observation.** `poisson`, `neg_binomial`,
  `binomial`, `beta_binomial` all apply `y.round().max(0.0)` before scoring
  (`obs_loglik.rs:174`, `obs_model.rs:118`). A per-1000 rate or a scaled series
  fed to a count likelihood is silently rounded — `0.4 cases` scores as `0`.
- **`bernoulli(p = …)` thresholds at 0.5.** Any observed value `> 0.5` counts as
  a success (`obs_model.rs:144`), so a count column accidentally bound to a
  Bernoulli stream reads as "detected" for every non-trivial value.
- **`beta(mean = …, concentration = …)` requires `x ∈ (0, 1)` strictly**
  (`obs_loglik.rs:353`). An observed proportion of exactly 0 or 1 — a common
  outcome for small denominators — scores `−∞` and takes the whole fit with it.
  Use `beta_binomial` with the denominator as `n` when the data is `k` of `n`.

Pairing guidance:

- **Incidence:** `neg_binomial` or `poisson`, with the reporting rate in the
  mean (`mean = rho * projected`). Support on ℤ≥0; overdispersion natural.
- **Prevalence, single compartment:** `binomial(n = N, p = projected / N)` when
  the denominator is known and observed; `poisson` for large `N`. `neg_binomial`
  is valid but its dispersion means something different here than it does for
  incidence.
- **Prevalence as a fraction** (projection ∈ [0, 1]): `beta` for a directly
  observed proportion, `beta_binomial` for a `k`-of-`n` count.

**A first incidence observation at the model origin is refused.** Incidence at
`t = t_start` has a zero-width accumulation window, so its expected count is
identically 0; a positive count against it scores `−∞`, which is
indistinguishable from a degenerate filter. The check
(`rust/crates/cli/src/util.rs:1473`) rejects it before the filter runs and names
three fixes: drop the origin row, date each row at the _end_ of its accumulation
window, or move the model origin earlier. A zero count at the origin is
consistent with the zero-width window and is allowed.

**The startup block prints the pairing — on three of the stage types.** `pgas`,
`pmmh`, and `nuts` stages print each stream's projection kind and likelihood
family before running, and warn when a `neg_binomial` is paired with a snapshot
projection:

```
── stage: post (method=pgas) ──
  2 observation streams: detection, weekly_cases
  observations (2 streams):
    ✓ weekly_cases     incidence(infection)         NegBinomial
    ✓ detection        prevalence(I)                Bernoulli
```

`camdl pfilter`, `if2`, `mh`, and the NLopt stages print only the bound-stream
line
(`pfilter: bound streams: detection(bernoulli), weekly_cases(neg_binomial)`) —
the projection kind is not shown there. Read the model, or run `camdl inspect`,
when you need to confirm a pairing on those paths.

---

## Appendix A: CLI Reference

Generated from `--help` at every level of the real command tree.

### Global options

Accepted on every subcommand:

```
--verbosity <LEVEL>   error/warn/info/debug/trace; overrides RUST_LOG (default warn).
                      --progress plain auto-bumps to info
--progress <MODE>     auto (default) | pretty | plain | none
--no-progress         shorthand for --progress none; wins over --progress
--no-ir-cache         recompile the .camdl instead of reusing ~/.cache/camdl/ir
                      (or $CAMDL_IR_CACHE_DIR)
--no-licm             disable loop-invariant code motion (on by default). Value-
                      preserving, but it changes the compiled IR, so a --no-licm
                      run re-keys the IR cache and run identity. = CAMDL_NO_LICM=1
-h, --help            print help
-V, --version         print version (top level only)
```

### The command tree

```
camdl
├── simulate MODEL              forward simulation
├── batch
│   ├── run FILE                sweep from a TOML manifest
│   └── status FILE             completion status of a sweep
├── fit
│   ├── run CONFIG              run the stages in a fit.toml
│   ├── summary FIT             R̂ / gate verdict / MLE table for one fit
│   ├── diff A B                compare two fit.toml configs
│   ├── table ROOT              one row per fit under a results tree
│   ├── new --from A DEST       derive a new fit.toml
│   ├── predict [FIT]           posterior-predictive (predicted vs observed)
│   └── methods                 list supported (algorithm, backend) pairs
├── pfilter MODEL               bootstrap particle filter at fixed θ
├── profile MODEL               profile likelihood over a parameter grid
├── survey MODEL                likelihood-landscape LHS diagnostic
├── data
│   └── split FILE              train / holdout split of a data TSV
├── list [ROOT]                 browse cached runs
├── show TARGET                 full metadata for one run
├── cat TARGET                  emit a run's trajectory or observation output
├── compare [PATHS…]            compare fits by prequential scores
├── label HASH LABEL            set a run's display label
├── check …                     parse + type-check (delegates to camdlc)
├── inspect …                   print model structure (delegates to camdlc)
├── render …                    LaTeX or display JSON (delegates to camdlc)
├── lineage
│   ├── realize EVENT_LOG       event log → line list
│   ├── tree LINE_LIST          line list → sampled transmission tree (Newick)
│   ├── sojourn LINE_LIST       dwell-time distribution in a compartment
│   └── cohort LINE_LIST        per-window event summary
├── mre
│   ├── fit CONFIG              bundle a fit's input closure
│   └── simulate MODEL          bundle a forward-simulation reproduction
├── docs [TOPIC]                embedded guides (offline, version-matched)
├── check-update                query GitHub for a newer release
└── dev                         developer & maintenance commands
    ├── reindex [ROOT]          rebuild <root>/index.json
    ├── eval MODEL --expr E     evaluate time-dependent expressions
    ├── compile …               .camdl → IR JSON (delegates to camdlc)
    └── doctest …               compile camdl blocks in Markdown (delegates to camdlc)
```

### `camdl simulate MODEL`

```
Model inputs
  --params FILE               parameter TOML (repeatable)
  --param NAME=VALUE          single override (repeatable)
  --param-vec PREFIX=FILE     indexed-parameter vector from a keyed TSV (repeatable)
  --table NAME=FILE           external table for table-lookup expressions

Scenario
  --scenario NAMES            named scenarios, comma-separated or repeated;
                              each is its own cell. Conflicts with --enable/--disable
  --enable NAME               enable an intervention (repeatable)
  --disable NAME              disable an intervention (repeatable)

Engine
  --backend gillespie|chain_binomial|ode        (default chain_binomial)
  --dt DT                     step for discrete-time backends (default 1.0)
  --integrator rk4|rk45       ODE method override; tolerances live in the model's
                              `simulate { integrator = rk45 { atol, rtol } }`
  --allow-degenerate-rates    restore legacy silent-zero on numerical collapse

Run window
  --to SPEC                   override the horizon (simulation.t_end). SPEC = a
                              model-time number, a date (date("YYYY-MM-DD") or bare
                              YYYY-MM-DD), or an observation anchor with an optional
                              offset ("last_obs + 8 weeks"). Anchored forms need
                              --fit for the observed times. A scenario declaring a
                              different horizon is an error
  --init-state FILE|fit       start from an inferred state at the last observation
                              time instead of init { }. The origin becomes t_start,
                              and must coincide with an output-emit time.
                              chain_binomial only. Two sources:
                                FILE — a `pfilter --save-final-state` particle
                                  ensemble at the filter's ONE θ. Its header carries
                                  the origin; replicate i restores particle row i, so
                                  --replicates must equal the row count. Conflicts
                                  with --draws
                                fit  — the --fit run's paired (θ_i, X_i(T)) posterior:
                                  draw i restores the terminal row of its OWN saved
                                  latent path under its OWN θ_i. Requires
                                  --draws posterior; runs over the subset of draws
                                  that have a saved path, and reports that count

Ensemble
  --seed N                    single seed (default 1; env CAMDL_SEED)
  --seeds SPEC                range "1:100" or list "1,2,42"; conflicts with --replicates
  --replicates N              stochastic replicates per parameter point
  --draws SOURCE              path to a params TSV | "uniform" | "prior" | "posterior"
  --fit PATH                  companion for --draws: fit.toml (prior), fit results
                              dir (posterior), or [fixed] backfill (file)
  -n, --n-draws N             number of draws (required for uniform/prior)
  --draws-out PATH            write the sampled parameter vectors as a TSV
  --parallel N                concurrent runs (env CAMDL_PARALLEL)

Trajectory output
  -o, --output FILE           plain-TSV mirror IN ADDITION to the store leaf
                              (env CAMDL_OUTPUT)
  --stdout                    stream to stdout, write no store leaf. Single-cell
                              only; conflicts with --seeds/--replicates/--draws,
                              -o and --obs*
  --output-every N            one row every N time-units, overriding output { every }
  --no-flows                  drop every flow_* column
  --columns COL,...           restrict trajectory columns to this allow-list
  --dates                     add a calendar `date` column (requires model `origin`)

Observation output
  --obs FILE                  synthetic observations, all streams, one TSV
  --obs-dir DIR               one TSV per stream
  --obs-only FILE             like --obs, suppress trajectory output
  --obs-only-dir DIR          like --obs-dir, suppress trajectory output
  --emit-every N | NAME=N     override `emit_schedule`, in model time units, for
                              every stream or one by its observation-block label
                              (repeatable; the two forms are exclusive). Only a
                              recurring `every N` cadence can be overridden; a
                              stream declaring `at [...]` is refused by name.
                              Refused outright when the run emits nothing

Other artifacts
  --quantities-out DIR        emit quantities { } as <dir>/quantities/<name>.tsv +
                              quantities.json. Refuses a multi-scenario run
  --event-log [FILE]          record the lineage event log into the run leaf; pass a
                              PATH to also mirror it. Single-run only
  --format parquet|tsv        event-log format (default parquet); --tsv is shorthand
  --reactive-log PATH         mirror the reactive firing log (leaf copy always written
                              when a reactive policy was active). Single-run only

Store
  --output-dir DIR            store root (default ./results; env CAMDL_OUTPUT_DIR)
  --force                     re-run even if cached output exists
  --label TEXT                display label, ^[a-zA-Z0-9 ,._-]{1,64}$
  --dry-run                   print the resolved run plan without simulating
  --cas                       no-op, accepted for compatibility (CAS is the default)
```

There is **no** `--sweep` on `simulate`; parameter sweeps are `camdl batch run`
(manifest) or `camdl fit run --sweep` (over fixed params).

### `camdl batch`

```
camdl batch run FILE
  --output-dir DIR            override output_dir from the manifest (env CAMDL_OUTPUT_DIR)
  --parallel N                override the manifest's thread count (env CAMDL_PARALLEL)
  --dry-run                   print the resolved sweep grid without running
  --force                     re-run even if output exists
  --allow-degenerate-rates    as on simulate

camdl batch status FILE       completed vs pending runs for a sweep manifest
```

There is no `--resume` flag: resume-by-cache is the default, and `--force`
overrides it.

### `camdl fit run CONFIG`

```
  --stage NAME                run only this stage
  --seed N                    RNG seed (default 1)
  --parallel N                Rayon thread cap; bit-identical regardless of value
                              (default 0 = all cores; env CAMDL_PARALLEL)
  --force                     re-run and overwrite stale cache
  --resume BASE_REF           extend a completed PGAS/PMMH stage from a base run
                              (run_id prefix or leaf path). Requires --stage;
                              conflicts with --force
  --sweep NAME=SPEC           Cartesian sweep over a fixed parameter (repeatable).
                              SPEC = V1,V2,... | lin(min,max,n) | log10(min,max,n)
  --label TEXT                display label
  --condition-from WHEN       burn-in / conditioning window; a model-time number,
                              a date, or "first_obs - 1 week". Re-keys the fit
  --allow-nonconverged-scout  proceed past a failed convergence gate

Chain initialisation
  --init MODE                 single | uniform | lhs | uniform_unconstrained (default)
                              | from_prior | from_posterior | from_mle | from_params
                              | survey_top_k
  --posterior PATH            companion for --init from_posterior
  --mle PATH                  companion for --init from_mle
  --params TOML               companion for --init from_params
  --survey-path DIR           companion for --init survey_top_k (requires --stage)
  --survey-top-k N            top-K count; defaults to the stage's `chains`

Post-fit audit
  --no-dt-check               skip the Richardson dt-convergence check at θ̂
  --dt-check-strict           strict warning thresholds (0.5 nats chain_binomial,
                              0.1 nats ode) instead of the routine 2.0 / 0.5
  --dt-check-halvings N       default 2 (dt, dt/2, dt/4)

Per-stage overrides (each requires --stage)
  --decibans-thresh DB        gate's inter-chain loglik-spread floor
  --cooling-target-iters N    IF2 cooling target
  --tempering B1,B2,...       parallel-tempering ladder; first value must be 1.0
  --max-tree-depth N          NUTS depth ceiling
  --trajectory-warmup N       CSMC-only sweeps before parameter updates
  --csmc-sweeps-per-nuts N    CSMC trajectory updates per parameter update
  --n-trajectories N          posterior trajectories saved
  --diagonal-mass             dense_mass = false (one-way)
  --no-nuts                   use_nuts = false — MH-within-Gibbs for θ|X (one-way)
  --no-adapt                  adapt = false — lock proposal SDs (one-way)
  --adapt-start N             step at which proposal-SD adaptation begins
  --rho F                     Crank-Nicolson correlation, [0, 1)
  --record-ancestry           record ancestor indices for smoothing paths
  --record-prequential        record per-step predictive samples for `camdl compare`
```

`--starts-from` no longer exists in any form; use `--init from_mle --mle <path>`
or the fit.toml key `init_mle`.

### `camdl fit` — the other subcommands

```
camdl fit summary FIT
  FIT is a handle: @label, a fit-level hash prefix, a results directory, or a fit.toml
  --stage STAGE               render only one stage's stanza
  --format text|json|md|latex (default text)
  --params-only               print only θ̂ as a flat params TOML (pipeable)
  --no-color                  disable ANSI colour (NO_COLOR is honoured regardless)
  --strict                    exit non-zero on provenance mismatch; auto-on when CI=true
  --exclude-chains IDS        recompute diagnostics over a chain subset (view only;
                              always warns; incompatible with --params-only)

camdl fit table ROOT          (see §13.1 for filters and formats)
camdl fit diff A B            (see §13.2)
camdl fit new --from A DEST   (see §13.3)
camdl fit methods             list (algorithm, backend) pairs with stability tiers

camdl fit predict [FIT]
  --fit FIT                   same handle grammar as `fit summary`
  --stream NAME               one logical stream, or an expanded leaf name
  --stage STAGE               use this stage's posterior cloud
  --scenario NAMES            prospective scenario overlay (repeatable). `fitted` is
                              reserved for the no-overlay row
  --enable / --disable NAME   ad-hoc overlay; conflicts with --scenario
  --sweep PARAM=GRID          vary a parameter across the posterior (repeatable).
                              GRID = list | lin(min,max,n) | log10(min,max,n).
                              Free-forward only; same param in a scenario is an error
  --horizon free_forward|one_step
  --n-draws N                 posterior subsample cap for both horizons (default 200)
  --seed N                    RNG seed for y_rep sampling (default 1)
  --exclude-chains IDS        drop 1-based chain ids before banding; always warns
```

### `camdl pfilter MODEL`

```
--particles N               required
--params FILE / --param NAME=VALUE / --table NAME=FILE
--scenario NAME | --enable NAME | --disable NAME
--dt DT                     (default 1)   --seed N (default 1)
--parallel N                Rayon threads, 0 = all cores (env CAMDL_PARALLEL)
--allow-degenerate-rates
--data [NAME=]PATH          observation TSV; repeatable. `--data PATH` binds the
                            single stream (or the one named by --obs); `--data
                            NAME=PATH` binds by name. Mixing the forms is an error
--obs NAME                  stream a bare `--data FILE` binds to
--fit PATH                  fit.toml supplying [data.observations]; consulted only
                            when no --data flag was given
--time-format auto|numeric|date
--pf-max-substeps N         deterministic per-call compute budget
--replicates N              independent filter runs (default 1)
-o, --output FILE           default stdout
--trace FILE                per-observation diagnostics TSV ("-" for stdout)
--pf-health FILE            per-obs ESS + Snyder τ², with the implied
                            particles-to-avoid-collapse estimate exp(τ²/2)
--save-final-state FILE     final particle states
--save-paths FILE           smoothing-distribution trajectory samples
--n-paths N                 how many, for --save-paths (default 1)
--save-filtering FILE       per-step particle states and log-weights
--save-prequential STEM     {STEM}.tsv + {STEM}.json one-step-ahead predictive
--no-save-samples           with --save-prequential, drop per-particle samples
```

`camdl pfilter` writes into the content-addressed store under
`$CAMDL_OUTPUT_DIR` (there is no `--output-dir` flag; `--output` is the TSV
mirror).

### `camdl profile MODEL`

```
--sweep NAME=SPEC           required, repeatable (2D+ grids). SPEC = V1,V2,... |
                            lin(min,max,n) | log10(min,max,n)
--particles N               required
--fixed NAME=VALUE          pin a parameter AND remove it from [estimate] (repeatable)
--fixed-file TOML           bulk form of the above (repeatable, later files win)
--iterations N              IF2 iterations per grid point (default 50)
--starts N                  independent starts per grid point (default 3)
--init MODE                 as on `fit run`; note the default here is `uniform`
--posterior / --mle / --params / (companions for the init modes)
--cooling F                 (default 0.95)          --rw-sd SPEC
--algorithm NAME            per-cell algorithm (default if2); pair with --backend
--backend NAME              chain_binomial (default) or ode
--pmmh-steps N / --pmmh-particles N / --pmmh-rho F      (PMMH only; 500/500/0.99)
--fit PATH                  fit.toml supplying priors, bounds and the fixed list
--suppress-warnings         silence the flat-prior fallback warning
--seeds SPEC                run the whole grid at each seed (multi-seed sensitivity)
--data / --obs / --time-format / --dt / --seed / --parallel / --scenario /
--enable / --disable / --table / --allow-degenerate-rates / --pf-max-substeps
-o, --output FILE           profile TSV (default stdout)
--label TEXT
```

### `camdl survey MODEL`

```
--fit FIT                   fit.toml supplying [estimate] bounds and [data].
                            Mutually exclusive with --estimate/--data
--estimate NAME=LO:HI       inline LHS bounds (repeatable)
--data PATH                 inline observation TSV
--fixed NAME=VALUE          pin a parameter not in --estimate
--scenario NAME
--n-points N                default 0 = auto: max(1000, 50·d²)
--eval auto|pfilter|simulate    auto picks pfilter when the model needs
                            OVERDISPERSION, simulate otherwise
--eval-particles N          (default 200)      --eval-replicates K (default 5)
--seed N                    (default 42)
--render                    also write an interactive landscape.html
--output DIR                output root (default ./results)
--label TEXT   --force   --parallel N
```

### `camdl compare [PATHS…]`

```
PATHS                       ≥2 when --config is absent. Each is a prequential.json
                            (or a stage dir holding one), read as-is, OR a fit
                            handle whose prequential is auto-derived from its θ̂
--config compare.toml       [[model]] entries; can also carry baseline/metrics/format
--baseline NAME             reference for Δ columns (default: argmax elpd)
--metric elpd,crps,pit_cov90
--format table|md|json      (default table)
--allow-mismatched-horizon  render even if T_score differs (Δ columns → '—')
--particles N               for derived prequentials (default 1000, applied uniformly)
--seed N                    for derived prequentials (default 1)
--exclude-chains [@FIT:]IDS per-fit (`@a:4`) or cohort-wide (`3,4`); mixing the two
                            forms is rejected. Always warns
```

Columns: `T_score`, `elpd`, `Δelpd`, `E_T` (= exp(Δelpd), the terminal e-value),
`se(Δ)` (paired standard error over pointwise differences), `crps`, `Δcrps`,
`PIT_cov90` (nominal 0.90; below 0.70 triggers an overconfidence warning).

### Browsing, labelling, data

```
camdl list [ROOT]
  --root DIR                  alias for the positional; wins when both are given
                              (default ./results; env CAMDL_OUTPUT_DIR)
  --model SUBSTR   --scenario NAME   --since 1h|30m|2d
  --kind sim|fit|profile|pfilter|survey|ensemble|all      (default all)
  --parent HASH               children of a specific parent run (8+ char prefix)
  --limit N (default 50) | --all
  --format human|json

camdl show TARGET             --root DIR   --format human|json
camdl cat TARGET              --root DIR   --stream NAME
camdl label HASH LABEL        --root DIR
camdl data split FILE         --at-time T | --fraction F, --time-col NAME,
                              --train PATH, --holdout PATH
```

### Compiler passthroughs

`check`, `inspect`, `render`, `dev compile`, `dev doctest` forward every
argument verbatim to `camdlc`; their own `--help` is short because the flags
belong to the compiler. `camdlc --help` is authoritative.

```
camdl check FILE.camdl [--no-dim-check] [--json-errors]
camdl inspect FILE.camdl [--summary|--dims|--compartments|--transitions|--tables|
                          --forcings|--ascii]
camdl render FILE.camdl [--format json] [--expand DIM]
camdl dev compile FILE.camdl [--set NAME=VALUE] [--json-errors] [--no-state-grad]
camdl dev doctest [--gate] FILE.md …
```

### `camdl lineage`

```
camdl lineage realize EVENT_LOG
  --identity-seed N           i.i.d. draw from P(identities | event log) (default 1)
  -o PATH                     line-list output (default line_list.<ext>)
  --format parquet|tsv        overrides the extension; --tsv is shorthand

camdl lineage tree LINE_LIST
  --scheme flat:RATE | stratified:idx=rate,...,default=rate   (default flat:1.0)
  -o PATH                     Newick output (required)
  --sample-seed N             (default 1)

camdl lineage sojourn LINE_LIST --compartment IDX [-o PATH]
  IDX is the integer compartment index, not the name. A summary always goes to stderr

camdl lineage cohort LINE_LIST -o PATH
  --event infection|<transition id>   (default infection)
  --window W                  window width (default 1.0)
  --align-first-event         align windows to the first matching event, not t=0
```

### `camdl dev`

```
camdl dev reindex [ROOT]      rebuild <root>/index.json from a fresh walk of every
                              run.json. The index is only a cache — run.json is the
                              source of truth — so this is optional; useful after
                              copying a results/ tree
camdl dev eval MODEL --expr "a,b"
  --params FILE / --param NAME=VALUE / --table NAME=FILE
  --from T (0) --to T (100) --every DT (1)   |   --at T1,T2,…  (mutually exclusive)
  -o FILE                     default stdout
camdl dev compile / camdl dev doctest      (see "Compiler passthroughs")
```

### Batch TOML stability

**Batch TOML is v1 and will change.** Its field names (`[config]`,
`[[scenario]]`, `[sweep]`) are standalone and pre-date the v2 run-system types
(`SimulateJob`, `SweepSpec`, `Seeds` in `rust/crates/cli/src/fit/config_v2.rs`).
A future version will align the schema with v2. **External tooling should not
assume the current field names survive unchanged.** Open an issue if you're
writing such tooling and need a migration window.

Sensitivity analysis (Sobol indices and similar) is not a camdl concern. Run
`camdl batch run` to produce the output tree, then compute indices with R's
`sensitivity` package or Python's `SALib`.

---

## Appendix B: Parameter Files

### B.1 params.toml — a point m ∈ M

A flat TOML file, one key per parameter, values numeric:

```toml
beta = 0.3
gamma = 0.1
sigma = 0.2
rho = 0.4
k = 5.0
N0 = 1000000
I0 = 10
```

Consumed by `camdl simulate --params`, `camdl pfilter --params`,
`camdl profile --fixed-file`, `camdl fit run --init from_params --params`, and
fit.toml's `[fixed] from_file`. `--params` is repeatable; later files override
earlier ones.

**Values must be numbers.** A unit-annotated string is a hard error, not a
silent parse:

```
$ camdl simulate model.camdl --params bad.toml     # gamma = "0.1 /day"
error preparing CAS: bad.toml:gamma: expected a number or table section, got String("0.1 /day")
```

**A `[provenance]` table is skipped**, so the `mle_params.toml` a fit writes can
be fed straight back in as a params file without its metadata leaking into the
parameter namespace.

**Table sections mangle to indexed names.** A `[<prefix>]` section's keys are
joined with an underscore, which is the ergonomic form for indexed parameters:

```toml
gamma = 0.1

[beta]
north_child = 0.5
north_adult = 0.3
south_child = 0.6
south_adult = 0.35
```

sets `beta_north_child`, `beta_north_adult`, `beta_south_child`,
`beta_south_adult`. Only one level of nesting is supported; a non-numeric leaf
errors naming the section and key.

Values are checked after all resolution: every parameter must be finite and
within its declared `in [lo, hi]`, with all violations reported together.

### B.2 Indexed parameter overrides

For a model with indexed parameters — `beta[region, age]`, `R0[patch]` — the
bulk-load form is `--param-vec PREFIX=FILE`, reading a headerless TSV of
`key<TAB>value`. **The key is the index suffix, not the full parameter name**;
the loader forms `<prefix>_<key>`:

```tsv
north_child	0.5
north_adult	0.3
south_child	0.6
south_adult	0.35
```

```bash
camdl simulate multi_index_beta.camdl --params p.toml --param-vec beta=beta_vec.tsv
```

For a multi-dimensional parameter the key is the **pre-joined suffix** in
declaration order (`north_child` for `beta[region, age]`), not separate columns.
Blank lines and `#` comments are skipped. An unknown resulting name is a hard
error that shows the mangling, so a file that already carries the prefix fails
loudly rather than silently doing nothing:

```
$ camdl simulate multi_index_beta.camdl --param-vec beta=beta_bad.tsv   # key `beta_north_child`
error: --param-vec beta: unknown parameter 'beta_beta_north_child'
```

Matching is by name; row order is irrelevant, and a file may cover any subset of
the family's cells.

**Precedence.** `--param-vec` shares tier 5 with `--param` — the top of the
resolution chain — so a scenario's `set`/`scale` does **not** override it. (This
is the sibling-of-`--param` rule, and it differs from the pre-resolver behaviour
where scenarios won.) The `--param-vec` unknown-name check runs during full
parameter resolution, which `--dry-run` short-circuits: a `--dry-run` invocation
will not report a bad key.
