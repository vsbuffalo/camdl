# camdl Run System Specification

**Version:** 0.5-draft\
**Date:** 2026-10-24

> **Scope:** This spec describes camdl's run system: forward simulation (single
> runs, multi-cell ensembles and `--sweep` grids), inference pipelines
> (`fit.toml`), parameter sweeps, predictive workflows, and the two
> content-addressed stores that hold every result together with its provenance.

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
fit.toml         → how inference RUNS (what to estimate, priors, data, one method)
```

The `.camdl` file owns structure. It also owns two things that are easy to
mislocate: parameter **priors** (`p ~ Normal(0, 1)` in the `parameters` block)
and named **scenario presets**. It does not own concrete parameter point values
outside a preset.

| Concern               | Declared in                                                       | Overridable by                                                                                                                   |
| --------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Model structure       | `.camdl` only                                                     | never                                                                                                                            |
| Parameter names/types | `.camdl` only                                                     | never                                                                                                                            |
| Parameter values (M)  | params TOML, `[fixed]`, a scenario's `set`                        | draw/sweep point, scenario, `--param`/`--fixed`                                                                                  |
| Scenarios (σ)         | `.camdl scenarios { }`                                            | CLI `--enable`/`--disable`                                                                                                       |
| Interventions         | `.camdl` only                                                     | scenarios and `--enable`/`--disable` toggle them                                                                                 |
| Priors                | `.camdl` `~`, `fit.toml [estimate.p.prior]`                       | fit.toml wins over the model                                                                                                     |
| Backend               | fit.toml `[method]`, CLI `--backend`                              | `--backend` on `simulate` and `predict MODEL`                                                                                    |
| Integration step `dt` | `.camdl` `simulate { dt }`, or written as `--dt` or `[config] dt` | only `simulate --dt`; elsewhere a written step must equal the model's, and a `--dt` beside a fit file its `[config] dt` (§1.3.1) |
| ODE integrator        | `.camdl` `simulate { integrator }`                                | `--integrator` on `simulate`, `predict MODEL`                                                                                    |
| Seeds                 | CLI, `fit_seeds`                                                  | `--seed` / `--seeds`                                                                                                             |
| Sweep / design        | CLI `--sweep` / `--sweep-file`                                    | never                                                                                                                            |
| Estimate vs fixed     | `fit.toml` only                                                   | `fit run --sweep` varies a `[fixed]` parameter                                                                                   |
| Which method runs     | `fit.toml [method]`                                               | never — a second method is a second file                                                                                         |

**The model file is self-contained for a single run** provided every parameter
gets a value from somewhere: a params TOML, a scenario, or a `--param`. A
`parameters {}` declaration never carries a value (names, kinds, bounds and
priors only), so a parameter with no value from any source is a hard error
naming the parameter and the sources that were consulted.

### 1.3 Precedence Chains

Settings resolve by one of two rules. A parameter value follows the last-wins
chain below: a later source replaces an earlier one, and the replacement is
reported on stderr. The integration step and the ODE integrator define which
model is integrated, and follow the rules of §1.3.1: on a command that fits or
scores data, a written step that differs from the model's is refused rather than
replacing it, and a flag that moves a replay of a stored fit off the fit's step,
backend or integrator runs with a note naming both.

Parameter resolution for the simulate-side commands (`simulate`, `pfilter`,
`profile`, `eval`) is a single function, `params_resolver::resolve_parameters`
(`fit run` calls it too), which is the only writer of
`model.parameters[i].value` outside the IR layer. It layers six tiers, last
wins:

```
1.  compiled-in value        (`p.value` already in the IR; see below)
2.  fit.toml [fixed]         (inference commands, and --fit on others)
3.  --params / --fixed-file  (each file layered in order)
3.5 draw row or sweep point  (automated M-layer variation)
4.  scenario                 (preset `params` + multiplicative `scale`,
                              or an inline ad-hoc `set`/`scale`)
5.  --param / --fixed        (the user's explicit assertion; highest: overrides
                              every value a selected scenario does not set or
                              scale; a collision with a scenario is refused)
```

Tier 1 is whatever value the compiled IR already holds for a parameter. A
`.camdl` never writes one (`parameters {}` declares names, kinds, bounds and
priors; `beta : rate in [0.001, 2.0] = 0.3` is a syntax error), and the compiler
emits every declared parameter as `estimated` (bounds or a prior, no start) or
`required` (neither), both without a number. Tier 1 is therefore empty for any
model compiled from a `.camdl`, except where `camdlc --set NAME=V` pinned the
parameter to a fixed constant at compile time (or the IR was written by hand).
When it does supply a value, the provenance label is `model_default`.

A generated draw row (`--draws prior|uniform`) carries only the parameters its
measure varies — the parameters with a prior, or the bounded parameters a
uniform sweep draws. Every other parameter resolves per cell through tiers 1–5,
so `--params` and each cell's own scenario reach it (gh#949).

The structural distinction between 3.5 and 5: a draw or sweep value is
_automated_ M-layer variation and is counterfactual-modifiable, so a scenario
patch (tier 4) overrides it. A `--param`/`--fixed` value is the user's explicit
assertion about this specific run and overrides every value a selected scenario
does not set or scale; a collision with a scenario is refused (below). A named
preset and an equivalent inline ad-hoc scenario resolve at the same tier, so
they are indistinguishable in the resolved value — only the recorded provenance
label differs.

**No replacement is silent.** Whenever a tier replaces, with a different value,
a value that a tier other than tier 1 supplied, the run says so on stderr — one
line per replacement, naming the parameter, both values and both sources — and
records it in each leaf's `run.json`
(`inputs.parameters_provenance.<name>.overrides`, beside the lines themselves in
`inputs.notices`; recorded, never hashed):

```
info: scenario `lockdown` replaced I0 = 5 (from p.toml) with 3
warning: scenario `lockdown` replaced I0 = 5 (from mle_params.toml, a fit's estimate) with 3
```

A fit config's `[synthetic] true_params` file is tier 5 for the forward run that
draws the synthetic dataset, so it replaces a value the `[synthetic] scenario`
sets. That notice names the file, not a flag:

```
info: [synthetic] true_params truth.toml replaced beta = 0.8 (from scenario `hi`) with 0.3
```

A replacement is a warning when the replaced value is a fit's estimate (it came
from a params file carrying a fit `[provenance]` block): the run then simulates
something other than what the fit estimated. Every other replacement is info —
it is one the command's own flags asked for (a later `--params` file, a
scenario, a sweep or draw, `--param`). Replacing a tier-1 value, and replacing a
value with an equal one, are not reported. A multi-cell run reports each
(parameter, replaced source, winning source) once, with the range of values when
they differ by cell.

**Precedence orders values; it does not grant permission.** Where a scenario
would discard a value the user wrote for the same parameter, or a value the user
wrote would discard a scenario's, the run is refused before anything is written,
rather than resolved:

- a `--param NAME=VALUE` or a `--param-vec PREFIX=FILE` row (`simulate`), or a
  `--fixed NAME=VALUE` (`profile`), on a parameter a selected scenario `set`s or
  `scale`s, directly or through a preset it composes (gh#571). Tier 5 would
  replace the scenario's value in every cell, so differently labelled scenarios
  would all run the flag's value. The error names the flag (for `--param-vec`,
  the file and the row), every colliding scenario and what it does, and where
  the value belongs instead: in the model's `scenarios {}` block (a `set` in the
  scenario that needs it, or a variant that `compose`s the colliding scenario),
  in a `--params` / `--fixed-file` file for a `scale` (a scale multiplies the
  file's value), or dropped. A scenario whose only action is a `set` to the
  flag's own value is not a collision and runs, as is a scenario that touches
  neither. A `--params` / `--fixed-file` file is tier 3, below the scenario, and
  is not checked: a scenario overriding a parameter file is the intended
  layering. `pfilter` refuses `--scenario` with `--param` at parse time
  (gh#194); `camdl predict` and `fit run` take no tier-5 flag;

- a scenario that `set`s or `scale`s a parameter a `--sweep`/`--sweep-file` grid
  or a `--draws <file>` varies (`simulate`, `camdl predict`, `fit run --sweep`;
  gh#572). Generated draws (`--draws prior|uniform|posterior`) are not checked:
  a scenario over them is the counterfactual those runs exist for;
- a scenario that moves a parameter off the value a fit.toml's `[fixed]` gives
  it, by `set` or `scale`: on `fit run` (the fit.toml's own `scenario`), and on
  `profile --fit` (the scenario from the fit.toml or the `--scenario` flag). The
  error names the parameter, both values, the scenario and the fit.toml, and the
  fix: give `[fixed]` the scenario's value to run under the scenario, or remove
  the parameter from the scenario. A scenario landing on the `[fixed]` value is
  not a collision and runs;
- a scenario that `set`s or `scale`s a parameter a fit.toml's `[estimate]`
  lists, directly or through a preset it composes: on `fit run` (the fit.toml's
  own `scenario`) and on `profile --fit` (the scenario from the fit.toml or the
  `--scenario` flag). A parameter is either estimated or given a value, never
  both: the estimate would discard the scenario's value. The error names the
  parameter, what the scenario does to it and the fit.toml, and the fix: move
  the parameter from `[estimate]` to `[fixed]` with the scenario's value to fit
  under it, or name a scenario that does not touch it;
- a scenario that `set`s or `scale`s a parameter `profile` estimates, with or
  without `--fit`. The estimated set is what `--rw-sd` selects: under
  `--rw-sd auto`, every parameter that is not swept, `--fixed`, in a
  `--fixed-file` or in the fit.toml's `[fixed]`; under `--rw-sd NAME=SD`, the
  names listed. A fit.toml's `[estimate]` need not list them. The error names
  the parameter, what the scenario does to it and `--rw-sd`, and the fix: list
  the parameter in a `--fixed-file` (held out of the estimated set; the
  scenario's value is still the one used, since `--fixed` on a scenario's
  parameter is refused), drop it from `--rw-sd NAME=SD`, or name a scenario that
  does not touch it. A scenario setting the swept parameter is not refused: the
  grid replaces it, reported at info;
- `[fixed] from_scenario` naming a scenario that `set`s or `scale`s a parameter
  `[estimate]` lists, directly or through a preset it composes: on every command
  that reads the fit.toml (`fit run`, `profile --fit`). The import would make
  the parameter both fixed and estimated. The error names the parameter, the
  scenario and the scenario's value, and the fix: define a scenario that does
  not set the estimated parameters and name it in `from_scenario`, or drop them
  from `[estimate]` to hold them at the scenario's value (a scale-only parameter
  has the first fix only: the import has no value for the scale to multiply).

A fit.toml naming one parameter in both `[estimate]` and `[fixed]` is refused by
every command that reads it (`fit run`, `profile --fit`); neither block wins.

On `profile --fit`, a `--fixed` or `--fixed-file` naming a parameter of
`[estimate]` holds it and takes it out of `[estimate]`, reported at info (§2.8);
a fit's `mle_params.toml` is such a file. A draw/sweep value never kicks a
parameter out of `[estimate]`, and a scenario touching an estimated parameter is
refused (above).

**Priors** resolve on their own three-tier chain (`fit.toml [estimate.p.prior]`
→ the model IR's `~` declaration → flat). For `camdl profile` the flat tier is a
fallback that is recorded but not warned about; for `camdl fit run` it is
reachable only by writing `prior = { flat = {} }` explicitly — an implicit
fall-through to a flat prior is a hard validation error before the fit starts,
and the resolved source per parameter is recorded in `fit.meta.json`.

#### 1.3.1 The integration step and integrator

The integration step `dt` is the substep length of the chain-binomial backend
and of the fixed-step ODE integrator; the ODE integrator (`rk4` or `rk45`) is
the method the `ode` backend integrates with. Both are keys of the model's
`simulate {}` block (language spec §18). How a value written outside the model
combines with the model's depends on what the command does with the result.

**Fitting and scoring against data** (`fit run`, `camdl predict MODEL`,
`profile`, `pfilter`). The model's `simulate { dt }` is the step. A step written
by the caller, a fit config's `[config] dt` or a `--dt` flag, must equal it and
each other; when the model declares none, the written step is the step, and 1.0
when nothing declares one. `fit run` reads both `--dt` and its fit file's
written `[config] dt`, and writes the resolved step into the problem's
`[config] dt`, so a run at another step is another problem (§2.2). `profile` and
`pfilter` read both `--dt` and the `--fit` file's written `[config] dt`;
`camdl predict MODEL` reads `--dt` only. Each disagreement is refused before
anything runs:

```
error: fit.toml: [config] dt = 0.25 is not the integration step: the model declares `simulate { dt = 0.5 }`, which every run of it integrates at. Drop [config] dt, or change the model's `dt`.
error: --dt = 0.25 is not the integration step: the model declares `simulate { dt = 0.5 }`, which every run of it integrates at. Drop --dt, or change the model's `dt`.
error: the --fit file's [config] dt = 0.25 is not the integration step: ...
error: --dt 0.25 and the --fit file's `[config] dt = 0.5` disagree; drop one.
error: fit.toml: --dt 0.25 and `[config] dt = 0.5` disagree; drop one.
```

Two commands read a fit file's observed window without fitting:
`camdl design
lhs --fit` and `simulate --fit` with an anchored `--to` or an
anchored model. Each reads the window at the fit's step by this rule, so a fit
file whose written `[config] dt` disagrees with the model's step is refused
there with the third message above. `simulate --dt`, when given, is the step the
window is read at.

The integrator on these commands is the model's. `camdl predict MODEL` takes
`--integrator`, which replaces the method and keeps the model's `rk45`
tolerances; `fit run`, `profile` and `pfilter` take no integrator flag. A fit
records the step it ran at in `fit.meta.json` (`dt`, §2.2.1).

**Replaying a fit** (`simulate --draws posterior|prior|<file> --fit`, and
`simulate --params` with a file carrying a fit's `[provenance]` block). The
replay runs on the fit's backend, step and integrator (§3.4). A params file's
block records the backend and the step; the integrator is the model's, since a
fit takes no integrator flag. A `--backend` or `--dt` equal to the fit's changes
nothing. A `--backend`, `--dt` or `--integrator` that differs from the fit's
runs, with one stderr note naming both simulators, and a `--backend` change
keeps the fit's step:

```
note: @r0 was fitted on chain_binomial at dt = 0.1; this replay runs on chain_binomial at dt = 0.5 (--dt as given), so the parameters drive a different simulator from the one they were estimated on
```

For a params file the note names the file in place of `@r0`. `--integrator` on a
replay whose backend is not `ode` is refused:

```
error: --integrator selects the ODE integrator, but this replay of @r0 runs on chain_binomial, which has none.
  Fix: drop --integrator, or pass --backend ode.
```

When `--fit` names a fit config rather than a stored run, the fit's step is
resolved from the config and the model by the rule above, so a disagreeing
`[config] dt` is refused there too. `camdl predict @run` replays at the fit's
step and integrator and takes none of `--backend`, `--dt` or `--integrator`: a
run handle with any of them is refused, naming the flags given.

**Exploratory simulation** (`simulate` without a fit). The step is `--dt`, else
the model's `simulate { dt }`, else 1.0; `--dt` replaces the model's step
without a refusal. A `--params` file carrying a fit's `[provenance]` block makes
the run a replay of that fit (above). `--integrator` replaces the model's method
and keeps its `rk45` tolerances.

`--integrator` is refused on a backend other than `ode` by every command that
takes it (`simulate`, `camdl predict MODEL`), since it would select nothing yet
change the run's identity:

```
error: --integrator selects the ODE integrator, but this simulation runs on chain_binomial, which has none.
  Fix: drop --integrator, or pass --backend ode.
```

`camdl predict MODEL` says "this prediction" in place of "this simulation".

### 1.4 Core Design Rules

**One job type behind the verb.** `camdl simulate` builds a `SimulateJob`
(`sim_job.rs`) and hands it to one engine (`engine::run_job`), which plans the
`param-point × scenario × seed-slot` grid and drives it through a `RunSink`;
`camdl predict` builds the same job for its free-forward replay. `simulate`
stores each cell in the run store and renders its exports from the stored cells
(§4.4). `SimulateJob` deliberately has no serde derive — the CLI parses its own
surface (clap args) and lowers to the resolved struct, so there is no second
wire schema to drift.

**Every parameter choice is accounted for.** In `fit.toml` the rule is
exhaustive and enforced: every model parameter must appear in exactly one of
`[estimate]` or `[fixed]`; a parameter in both, in neither, or in
neither-and-not-in-the-model is a hard error before the fit starts. On the
simulate side the rule is weaker: the resolver walks the tiers above and errors
only if _no_ tier supplied a value. Because a `.camdl` declaration carries no
value, a simulate-side run needs a value for every parameter from a params file,
a scenario, a draw or sweep, or `--param`. An `[estimate]` parameter the model
gives no value is seeded with its `[estimate].start`, or a draw within its
bounds when the entry has none, and recorded as `estimate_start` or
`estimate_start_drawn`: a reference point, not any chain's start, which
`init_provenance` records. `fit run` and `profile --fit` seed through the same
function, so `beta = {}` means the same in both; a profile's swept parameter
takes its value from the sweep and needs no start. A refusal names the flags the
command has (`--params`/`--param` on `simulate`, `pfilter` and `eval`;
`--fixed-file`/`--fixed` on `fit` and `profile`).

**Sweeps are orthogonal to everything.** A sweep is "run this thing at multiple
parameter values," and it works on both sides. `simulate --sweep NAME=GRID`
(repeatable) or `--sweep-file FILE` varies a model parameter over a
deterministic grid — the design axis (§5.5). `fit run --sweep NAME=SPEC` varies
a `[fixed]` parameter, where SPEC is `V1,V2,…` | `lin(min,max,n)` |
`log10(min,max,n)`; sweeping a parameter that is in `[estimate]`, or not in
`[fixed]`, is an error. Sweeps compose with scenarios and seeds by Cartesian
product, provided the scenario does not touch a swept parameter (that is refused
on both sides; see §1.3 and §3.6).

**Draws and sweeps are different operations.** A sweep is a deterministic grid
the user designed; draws are samples from a distribution (posterior, prior, or
uniform over bounds). They have different provenance and different downstream
semantics, so they are separate variants of a sum type — `ParamSource::Point`,
`::Sweep`, `::Draws` — never conflated. `ParamSource::Draws` further records
whether the rows came from a user-authored file, because a scenario that patches
a parameter the file also supplies is refused for an explicit file (the scenario
would override that column in every draw), while over generated draws it is the
intended counterfactual and the scenario's value applies.

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
└── camdl-store/     # the run store (§4.4, §2.2, §2.8)
```

Only `camdl-store/` has meaning to camdl. `camdl simulate`, `camdl pfilter`,
`camdl fit run` and `camdl profile` store their results in the run store, and
`camdl list`, `show` and `cat` read it (§4.5). The store's layout is in §4.4,
for fits §2.2 and for profiles §2.8.

### 2.1 The run store's root

Every command finds the run store's root by one rule: `--store DIR`, else the
`store` key of the fit file the command reads (`fit run FIT.toml`, or
`--fit FIT.toml` on `simulate`, `pfilter` and `profile`; relative to the fit
file, §6.2), else `CAMDL_STORE`, else `./camdl-store`. A command that reads no
fit file, or is given a stored fit's handle rather than a file, skips the second
step. So a fit file's runs, and the results of every command given that file
with `--fit`, are in one store.

The root is not part of any identity: the same inputs stored under two roots
have the same run id.

### 2.2 Fit Result Layout

`camdl fit run` stores a fit in the run store (§4.4) as two kinds of result. A
**fit problem** (kind `fit`) holds what the fit is fitted to and how the problem
is posed, and is written once. A **fit run** (kind `fit_run`) is one run of one
method at one seed on that problem, stored inside it and written in place while
it works. The layout is that of `docs/dev/proposals/2026-11-20-run-store-api.md`
("Worked example: `camdl fit run`"). One IF2 fit with `holdout_after`:

```
<root>/<build>/
  compiled_model/<model stem>-<12 hex>/         # the compiled model, stored once
    model.ir.json …
  quantities/<model stem>-<12 hex>/             # the model's quantities and contrasts, stored once
  fit/<model stem>-<12 hex>/                     # the fit problem
    identity.json  run.json
    problem.toml                                 # the resolved problem, in fit.toml syntax
    partition.json                               # which rows trained
    inputs/<source>.<ext>                        # each data file's training lines
    inputs/tables/<name>.<ext>                   # each [data.tables] file, as read
    held_out/<stream>.tsv                        # held-out rows' dates, no values; only when rows are held out
    synthetic/truth.toml                         # a [synthetic] problem's true parameters
    model.camdl                                  # the model file, under the name it was given (model.ir.json for IR)
    runs/<method>-<12 hex>/                      # one group per set of method settings
      seed_<n>-<12 hex>/                         # one fit run
        identity.json  run.json
        fit.toml                                 # the fit file this run was given, byte for byte
        method.toml                              # the run's [method] and the run settings of [estimate]
        run_inputs.json                          # what the run was asked to do and what it found
        fit.meta.json                            # the problem's provenance as the run resolved it (§2.2.1)
        execution.json  events.jsonl  progress.json
        inputs/starts.<ext>                      # a --starts file named directly
        fit_state.toml  mle_params.toml  chain_0/ …   # the method's results, below
        predict/<mode>-<12 hex>/                 # `camdl predict` output (§2.2.2)
```

`<build>` is the camdl version and 12 hex digits of the ambient key, as for
every result in the run store (§4.4). Each folder name below it is a label
followed by 12 hex digits of that result's or group's key; the label is not part
of the identity. The problem's label is the model file's stem, slugged
(`sir-beta.camdl` gives `sir_beta`), the method group's is the algorithm's name,
and a run's is `seed_<n>`, the seed typed. A sweep point (§2.3) and each
`[synthetic]` dataset is a problem of its own, in a sibling folder with the same
label.

**What the problem's identity holds.** The compiled model it depends on; each
data file's training lines; each stream's held-out rows, by date and without
their values; the partition rule; the `[data.tables]` files; `[estimate]`'s
estimated set with its bounds and priors; every `[fixed]` value after
resolution, a sweep point's included; `simplex_groups`; the settings (`dt`, the
step resolved from the model, `--dt` and `[config] dt` by §1.3.1,
`allow_degenerate_rates`, `allow_forcing_extrapolation`); the scenario request
(`scenario`, `enable`, `disable`) as given; `ic_free`; and, for a `[synthetic]`
problem, what generated its data: the true parameters file's bytes, the
dataset's one seed and the block's other settings. Two fit files that pose one
problem and differ in `[method]` share one problem folder. The steps, in
`[method.perturbation]` or `[method.proposal]`, belong to the method, and
`start`, though written under `[estimate.<parameter>]`, belongs to where the
chains start, so two runs that differ only in them share a problem.
`[synthetic]` and `--emit-every` generate training rows, which the problem keys
like any data. A `[data.holdout]` table's values enter no fit; its rows' dates
are held-out rows like a cutoff's.

Data enter by **content**, not path. A change to any column within the training
lines re-keys the problem, and with `holdout_after` the lines past the cutoff
are not in it, so a change to a held-out value does not.

**What a run's identity holds.** The problem it depends on; the method's
settings without its length, which for `if2` include every estimated parameter's
resolved perturbation (its SD in z and whether it is perturbed at t = 0 only)
and each simplex member's SD, and for `pmmh` and `mh` every estimated
parameter's resolved proposal SD in z; the larger compile with the state
Jacobian when the method needs it (`nl-lbfgs` on the ODE backend), the process
seed, where the chains start, the length (PGAS `sweeps`, PMMH and `mh`
`iterations`), and for a resume the base run (below). Where the chains start is
the `starts` rule, what it reads, and the `start` values `[estimate]` writes. A
step is keyed as resolved, so a step written at its default value and a step
left out give one run, and two values that differ in a bit give two. A sourced
`starts` (`from_mle`, `from_posterior`, `from_params` on a stored fit) holds a
dependency on the file it reads from the source run, so refitting the source
with identical inputs leaves the dependency, and every run started from it,
unchanged; a `--starts` file named directly is stored in the run as
`inputs/starts.<ext>` and keyed by its bytes.

**The files a problem and a run hold.**

- `problem.toml` is the problem in `fit.toml` syntax, written from the problem's
  identity alone and parsed back by `fit run`'s own parser before it is stored.
  `[model] camdl` names the stored model file, `[data.observations]` the stored
  training files, one per source, never a wide `file`, and `[data.tables]` the
  stored tables. It holds no `holdout_after` and no `[data.holdout]`, every
  `[fixed]` value resolved (a `from_file` read in, a `from_scenario` expanded),
  `[config]` with the step the one dt rule resolved, and `[simulate] from`. A
  `[synthetic]` problem's file also holds `[synthetic]` with
  `true_params = "synthetic/truth.toml"` and its dataset's one seed in
  `sim_seeds`.
- `partition.json` is the training partition (§6.2): `rule_version`; `rule`, one
  of `{"kind": "all"}`, `{"kind": "cutoff", "at": {"t", "date"}}` and
  `{"kind": "separate_files"}`; `extent`, `{"t", "date"}`, the latest end among
  the training rows; and `streams`, per stream `{"stream", "rows", "digest"}`:
  rows `[0, rows)` trained, and `digest` is the row digest of exactly those
  rows.
- `inputs/<source>.<ext>` is the source's data file with only the lines that
  hold a training row of one of its streams and the lines that give no stream a
  row, every column kept; when every stream trains on all its rows it is the
  source's bytes. Binding the stored bytes must give each stream exactly its
  training rows, by count and row digest, or the problem is refused.
- `held_out/<stream>.tsv` lists each held-out row without its value: `file`
  (`training` for a row past the cutoff, `holdout` for a row of a
  `[data.holdout]` table), `position` in that file, `label`, `measures`
  (`instant` or `period`) and `start`/`end`.
- A run's `fit.toml` is the file it was given, comments included. Its
  `method.toml` is the `[method]` table after the command line's method flags
  are applied, with every estimated parameter's resolved step written into
  `[method.perturbation]` or `[method.proposal]`, followed by an
  `[estimate.<parameter>]` table with its `start` for each parameter whose start
  the file writes; with `problem.toml` it parses to the configuration the run
  computed. `run_inputs.json` records the method, backend, seed, chain count,
  starts rule and source, binomial sampler, best chain and log-likelihood, each
  parameter's step (§9.5), and resume lineage; every reader of a run reads its
  method and backend there.

A reader of a stored fit reads the problem through these files and the stored
compile, never through the user's fit file, model or data: an edit to any of
them after the fit does not reach a replay.

**A rerun.** `fit run` reuses a completed run of the same identity
(`cache hit — reusing <folder>`) and computes only the problems and runs the
store does not hold. `--force` recomputes the run: the stored one is moved to
quarantine, its predictions with it.

**A run's result, apart from its executions.** A run's _result files_ — the
files its `run.json` lists with role `result` — are a function of its
computation. An _execution_ is one attempt to perform that computation: the
first run, a run killed and re-run, a run on another host. Everything about an
execution rather than its result is kept in files of another role (§10.6), in
three per-attempt records, and a fourth for an NLopt run:

| file             | what it records                                                                                                                                                                                                            |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `execution.json` | the execution record (`camdl.execution/v1`): `started_at`, `finished_at`, the method's `wall_time_seconds`, `pid`, `host`, `threads`; PMMH and `mh` add `cost`, PGAS `sampling_seconds`; a resumed leaf `earlier_segments` |
| `progress.json`  | the liveness heartbeat, rewritten on a wall-clock timer while the method runs (§10.10)                                                                                                                                     |
| `events.jsonl`   | what the run reported, one line per event (§10.11); not the lineage event log of `simulate --event-log`                                                                                                                    |
| `opt_trace.tsv`  | an NLopt leaf's live search trace: each search's running best log-likelihood, a row every 25 evaluations or 2 seconds (§10.13)                                                                                             |

A fit run's method works in a directory of its own outside the store. While it
runs, the run's folder holds its record, `fit.toml`, `method.toml`, and the
method's `progress.json` and `events.jsonl`, passed on to the folder about every
half second; the method's result files, `execution.json`, `run_inputs.json` and
`fit.meta.json` are written into the folder when the method ends, or when camdl
stops the run on an error. A run whose process is killed therefore leaves no
chain files and no `execution.json` in its folder. A re-run that reclaims a dead
attempt's run moves that attempt's `run.json` (as `attempt.json`) and its
records into `.attempts/<n>/` before recomputing, so a leaf keeps one set of
records per attempt; the set at the leaf root belongs to the attempt that wrote
the result. `fit summary`'s ESS per second and the end-of-run `complete in` line
read wall time from `execution.json`; a leaf without one reports none.

Byte reproducibility — two executions of one computation writing identical
stored bytes — is promised where it is tested: on the committed fixture
(`rust/crates/cli/tests/fit_leaf_byte_reproducible.rs`), PMMH, PGAS, IF2 and
`nl-sbplx` each write identical manifests (path, length and SHA-256 of every
file, method leaf and child leaves) when run twice uninterrupted into fresh
stores, when run into a store at another location relative to the model and
data, and when killed mid-method and re-run. A fit extended with `--resume`
(below) is not byte-identical to an uninterrupted fit of the same length: on the
same fixture, PMMH extended from 300 to 500 steps and PGAS from 200 to 300
sweeps reproduce the base segment's rows exactly and diverge after the resume
boundary, because the chain state a resume continues from does not carry the
random-number stream. An extension is statistically, not byte, reproducible
against the uninterrupted fit. These tests are regression evidence on one model,
not proof: a model that reaches a different parallel reduction, scheduling path
or platform mathematics routine may not reproduce its bytes, and the guarantee
is stated for one machine image.

**A resumed or extended fit.** PMMH and PGAS can be extended (`fit run` refuses
`--resume` for every other method, `mh` included): `fit run --resume <base>`
with a longer `iterations` / `sweeps` continues each chain of `<base>` from its
checkpoint. Four things are distinct:

- the _computation's identity_ includes its target length: the extended fit is a
  different computation, with its own `run_id` (the length is in the run's
  identity, and the base run is a dependency of it over the files a resume
  carries: each chain's `resume_state.bin`, `trace.tsv` and `trajectories.tsv`).
  It is stored beside the base, under the same method group;
- a _checkpoint_ — `chain_N/resume_state.bin`, the chain state at the end of the
  base — is a state a longer computation can start from. A resume is admitted
  only when the extension's identity, with its length and its base taken out,
  equals the base's, and the base came from the same build, so only the target
  may differ. A checkpoint is never served as a completed result: a resume
  toward a target the base has already reached is refused before any leaf is
  claimed, since the leaf it addressed would hold the base's whole, longer chain
  under the shorter computation's identity; at the base's own target, the base
  is the result;
- a _completed result_ at a target length is an ordinary leaf. Extending it
  reads the base and writes a new leaf: the base's leaf is unchanged, is found
  under its own `run_id`, and reports its own length
  (`rust/crates/cli/tests/fit_extension_is_its_own_computation.rs`);
- the _executions_ that produce or extend them are recorded per leaf, in
  `execution.json`; a resumed leaf's lists the earlier segments of its chain
  with their wall times, which sum to the wall time behind its draws.

**Leaf contents by method.** An IF2 leaf holds `fit_state.toml`,
`mle_params.toml`, `final_params.toml`, `chain_starts.tsv`,
`chain_evaluations.tsv`, `diagnostics.tsv`, `diagnostics.json`, and per chain
`chain_N/parameter_traces.tsv` + `chain_N/final_params.toml`. A PGAS or PMMH
leaf holds `fit_state.toml`, `chain_starts.tsv`, `draws.tsv` (the thinned
post-warm-up cloud), `diagnostics.json`, and per chain `chain_N/trace.tsv`,
`chain_N/resume_state.bin`, and — when posterior trajectories are requested —
`chain_N/trajectories.tsv` with a `chain_N/trajectories.json` manifest.

`mle_params.toml` is written params-first with a trailing `[provenance]` table,
so a bare `toml` reader still sees the parameters at the top level. The block
records `camdl_version`, `content_hash` (a tamper hash over the numeric values
only, at 12-decimal precision, so editing a provenance field does not invalidate
it), `fit_hash` (the problem's id), `backend`, `dt`, `model_identity`, `seed`,
`method`, `chain` (counted from 0, §6.8), `log_likelihood`, `loglik_sd`,
`n_particles`, and optional `ess_mean`/`ess_min`. It records the model by
content and holds no time, no path and no data-file digest: the file is a result
of the run, a function of its identity, and the identity holds the training
lines and the held-out dates rather than whole data files. The whole-file
SHA-256 of each data file the fit read is in the run's `fit.meta.json`
(`data_hashes`), a provenance file. The model's file is the problem folder's
`model.camdl` (or `model.ir.json`). For IF2, `log_likelihood` is the loglik-eval
re-score of the winning chain, so `loglik_sd` is that re-score's Monte Carlo
standard error (the `±` on the stderr line), `n_particles` its particles per
replicate (not the IF2 swarm's size), and `ess_mean`/`ess_min` its effective
sample size; an ODE-backend NLopt fit has no particle filter and records
`loglik_sd = 0` and `n_particles = 0`. `camdl simulate --params` reads
`backend`/`dt` back and replays the fit on them, with a note when a flag moves
the run off them (§1.3.1), and refuses the file when its parameter values no
longer hash to `content_hash` (they were edited after the fit), naming the file:
re-run the fit, or remove the `[provenance]` block to run the edited values as a
plain params file.

`camdl fit summary RUN --point posterior-mean` (or `posterior-median`) prints a
file of the same form for a sampler run (MH, PMMH, PGAS). Each estimated
parameter's value is the arithmetic mean (or the median, interpolated between
the two middle draws when their number is even) of its column of the run's
`draws.tsv`, the retained chains pooled, and each fixed parameter's is the value
the fit held it at. The statistic is taken on the natural scale `draws.tsv`
holds; a mean taken on a transformed scale differs (the exponentiated mean of
log β is the geometric mean, below the arithmetic mean unless every draw is
equal). The point is not one a chain visited and was not scored.

The chains pooled must agree, since a mean of two modes can fall between them. A
chain that took no step after burn-in is refused, naming it and the
`--exclude-chains` that leaves it out; `--allow-nonconverged-source` does not
lift this. A convergence verdict that is not converged is refused by the rule a
`from_posterior` source is judged by (§6.6): the run's stored verdict, or under
`--exclude-chains`/`--chains` the R̂ of the chains kept.
`--allow-nonconverged-source` summarises such a run with a warning. A verdict
that cannot be read is refused in either case.

Its `[provenance]` block holds `camdl_version`, `content_hash`, `fit_hash` (the
problem's id), `backend` (the run's own), `dt`, `model_identity`, `seed` and
`method` as above, then `statistic` (`posterior_mean` or `posterior_median`),
`n_draws`, `run_id` (the fit run's full id), `excluded_chains` (when chains were
left out) and `nonconverged_allowed = true` (when the override was used), in
place of `chain` and the log-likelihood fields. A block holding fields of both
kinds is refused. Of the simulator the block records the backend and `dt` only:
`simulate --params` replays the file on them with the model's integrator,
default settings and its own binomial sampler, checks `content_hash` as it does
an optimizer's file, and advises printing the summary again when the values were
edited. A run fitted with `allow_degenerate_rates`,
`allow_forcing_extrapolation` or an integrator other than the model's is
refused, naming the setting. `camdl predict MODEL --params` predicts at the file
(§4). An optimizer run is refused, naming `--params-only`, which prints its
winner.

#### 2.2.1 `fit.meta.json` — the run's record of its problem

Each fit run holds a `fit.meta.json`, a provenance file recording the problem as
that run resolved it. It is **derived provenance, never a source of truth**:
everything in it is a readable projection of inputs already in the problem's and
the run's identities, written after identity is fixed and never fed back into
any hash. A reader that needs the problem reads `problem.toml`, `partition.json`
and the stored inputs (§2.2). Its keys, top-level and inside each map
(`data_hashes`, `fixed`, `parameters_provenance`, …), serialize in sorted order,
so two identical runs produce byte-identical files and a diff of two runs'
metadata is meaningful.

Fields: `schema_version` (5), `store_layout`, `model_identity` (the hex
structural model hash), `fit_toml_hash` (SHA-256 of the fit file's raw bytes,
the run's `fit.toml`; provenance only, and no handle is looked up by it),
`data_hashes`, `holdout_hashes`, `table_hashes`, `estimated`, `fixed`,
`partition` (the run's parameter roles, below), `resolved_priors` (one
`{param, source, prior, bounds}` entry per estimated parameter, below, plus
`overridden_model_prior` when the fit file's prior displaced a model `~` prior;
recorded whichever method ran, since the resolution is a fact about the
problem), `parameters_provenance` (per parameter: resolved value, the tier that
supplied it, and its role), `training_partition` (the problem's
`partition.json`, §2.2; `date` is present when the model has an `origin`),
`forcing_interpolation` (absent when empty; one
`{forcing, method, cut, rows_after_cut}` entry per table forcing whose values
before the cut, the training extent, depend on `rows_after_cut` rows after it,
which `fit run` also says once as a note: "values before the cut depend on N
rows after it"; see "Table forcings across a cut" below), `dt` (the integration
step the run ran at — the model's declared `simulate { dt }`, else the config's
`[config] dt`, else 1.0) and `integrator` (the ODE integrator the model
declares, `{"method": "rk4"}` when it declares none), which a replay of the
fit's parameters runs on by default (§3.4), `calendar` (the model's calendar
axis, §2.2.2's block, so the fit's model times can be dated without the model;
required), and `schema` (the observation/dimension schema below).

**Table forcings across a cut.** A fit's cut is its training extent, and a
forecast's is its origin. A table forcing can give values before the cut that
depend on table rows after it, so a fit or the filtering before a forecast's
origin reads information from after the cut. camdl records such a forcing and
says so; it refuses nothing and changes no value, and the record is provenance,
in no identity. It is judged on the table's argument `t − lag`, over the model
times from the run's start to the cut:

- a cubic-spline table (`method = spline`) depends on every row, so on every row
  after the cut whenever it has one;
- a linear table depends on the first row after the cut when the cut falls
  strictly between two rows and the run reads an argument between the last row
  before the cut and the cut;
- a step table (`method = constant`) never does;
- a lag, which is never negative (language spec §7, "Lagged forcing"), moves the
  arguments read before the cut earlier, and a lag that is a parameter is judged
  as a lag of 0, the value whose arguments reach closest to the cut.

The note reads "forcing `pop` (linear table): values before the cut (t = 24, the
training extent) depend on 1 row after it". A fit records it in `fit.meta.json`
(`forcing_interpolation`); a forecast records it in `forecast.json`'s
`covariates.interpolation` (§2.2.2), beside any `--covariate` statement.

`resolved_priors` holds, per estimated parameter:

- `source`, the tier that supplied the prior:
  `fit_toml, model_ir, flat_explicit, flat_fallback`. `flat_fallback` appears
  only on a problem no posterior sampler can run on, since a sampler refuses an
  implicit flat prior before it starts.
- `prior`, the distribution with its arguments as numbers, in the form a fit
  file's `[estimate.<param>] prior` takes:
  `{"log_normal": {"mu": -1.0, "sigma": 0.5}}`,
  `{"uniform": {"lower": 0.1, "upper": 0.9}}` (a `prior = { uniform = {} }`
  resolved against the bounds), or `{"flat": {}}`. A hierarchical prior is
  `{"hierarchical": {"kind", "args", "pool_over"}}`, as the compiled model
  records it, each argument an expression over other parameters.
- `bounds`, `[lower, upper]`, the interval the run searches the parameter in and
  keeps it inside: the fit file's `[estimate]` bounds when it gives them (which
  must lie inside the model's declared `in [lo, hi]`), else the declared range,
  else the range of the parameter's kind. An infinite end is `"inf"` or
  `"-inf"`.

`schema_version` is 5, and every reader refuses any other value, naming it,
rather than reading an older record's fields as if they meant the same thing.

`data_hashes` holds, per observation source the fit binds (the `from <label>` a
data file is bound to, which is the stream's name for a plain stream), the full
SHA-256 of the data file bound to it, as the fit file names it and before the
training cut, so under `holdout_after` it is not the hash of the problem's
stored `inputs/` file. The `[data] file = "..."` shorthand records the one file
under every source it binds; `[data.observations]` records each file under the
key written. `holdout_hashes` records each `[data.holdout]` file the same way
and is omitted when there is none; `table_hashes` records each `[data.tables]`
file and is omitted when there is none. Under `fit run --sweep` each point's run
records its own point's value of the swept parameter in `fixed`, as its
`partition` does.

`partition` is the authoritative record of the run's roles. Its `parameters` map
holds, per parameter, `role` (`estimated` or `fixed`), `declared_by`
(`fit_estimate` or `fit_fixed`), and for an estimated parameter its `support`,
its `search` box, the `transform`, and the `prior` that resolved (`dist` and
`from ∈ {fit_toml, model_ir, flat_explicit, flat_fallback}`). `estimated` and
`fixed` are generated from it. An IF2 fit records its priors in both places even
though the method does not read them.

**A fit file never names a run.** One unchanged fit file makes several runs
whenever something outside its text changes: the data, its seeds, a sweep, or
the build. A command that needs one run (`fit summary`, `camdl predict`,
`compare`, `simulate --draws posterior --fit`, `--starts from_mle=…`,
`fit run --resume`) refuses a fit file and lists the runs the file has made,
each with its id and folder:

```
error: fit_pmmh_all.toml is a fit configuration, and a configuration never names a run: one file makes several runs whenever its data, its seeds, a sweep or the build change.
    9cd433e2dedb  camdl-store/0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-797c0b53281a/runs/pmmh-85f985a8c2a2/seed_1-9cd433e2dedb
  Name one by its id or folder, or by `@label` (`fit run --label`).
```

A command that needs only the problem (`pfilter --fit`, `design`,
`profile --fit`) accepts a fit file, resolved in process, or a stored fit.

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

#### 2.2.2 `camdl predict` predictive artifact

`camdl predict RUN` resolves the run handle — `@label`, a run id prefix or a
run's folder — to one run. A fit file is refused with the runs it made listed
(§2.2.1). It stores one `prediction` result inside that run, at
`predict/<mode>-<12 hex>/`, holding two tidy, plot-ready families of files plus
a JSON manifest for each that records the run's full `run_id`. Two runs of one
problem keep separate predictions, and each file is attributable to the run it
replays.

A run of an optimizer (IF2, or an NLopt method) has one best-fit point and no
posterior, and is predicted from that point once: its winner, as
`fit summary --params-only` prints it, labelled `plug_in` as a fixed point's
bands are. `--n-draws` above one is refused for it, since there is one point.
Its training rows are the run's problem's, so its scores are conditioned as a
sampler's are: `in_sample`, or `hold_out_tail` from the training extent when
scored rows end after it (§2.2.4); never declared or unknown. Its score records
`parameters = "winner"`, which a fixed point's `fixed_point` does not share. A
sampler run whose draws are not yet written is refused, naming its method.

`camdl predict RUN --model FILE` evaluates the quantities and contrasts of the
model file `FILE` in place of those of the model file the run's invocation
recorded. `FILE` is compiled and compared with the model the fit ran on as
`simulate --fit` compares a model (§3.4): a file that differs is refused, and
one that differs only in its `quantities {}` or `contrasts {}` is said on
stderr. The fit is read from the store and not computed again. `FILE`'s
quantities and its model source are stored as any compile stores them, and the
invocation records `FILE`'s model source as the source of the fit's compiled
model.

`camdl predict MODEL`, with a `.camdl` or `.ir.json` model file in place of the
handle, predicts at one point: the model's values, each `--params` file, then
each `--param` (§1.3, without a fit config's tier). `--data` is required and
bound as `pfilter` binds it; `--holdout-after` cuts each bound file as a fit
problem's `inputs/` are cut (§2.2.1): every row is parsed to find the cut, so a
malformed held-out row is refused, and a held-out row is then neither stored nor
used (a `last_obs - …` cut resolves over the whole files, held-out rows
included, as a fit's does); `--table`, `--backend`, `--integrator`, `--binomial`
and `--allow-degenerate-rates` are the numerics `simulate` resolves, and `--dt`
must agree with the model's declared step (§1.3.1). The point is repeated
`--n-draws` times (default 200), each copy under the seed a posterior draw at
its place gets, and every mode pools over the copies as it pools over a
posterior, so the bands carry no parameter uncertainty. The prediction is stored
at `prediction/<model>-<12 hex>/<mode>-<12 hex>/`, grouped by the compiled
model. Its identity is the compiled model rewritten from the data (anchored
times, bound tables, `--integrator`), every parameter's value, the numerics
(backend, step, binomial sampler, `allow_degenerate_rates`), the cut in model
time, each stream's conditioned rows as a stored data table, the model's
quantities, and the same options as a posterior prediction's; a held-out row's
value is in none of them. `--no-licm` changes the compiled model and so re-keys
it. A posterior flag (`--by-chain`, `--exclude-chains`, `--chains`,
`--realised-design`, `--model`) is refused with a model file, and a fixed-point
flag with a handle. A window-column stream with no emit schedule has no forecast
target, since no held-out row is kept, and is refused for a forecast. Contrasts,
which fork from a fit's saved smoothing paths, are skipped with a notice.

Both forms take `--store DIR` (the store a fixed-point prediction is stored in
and a run handle is looked up in, over `CAMDL_STORE`; a folder handle names its
own store), `--label`, `--force` (compute a stored prediction again) and
`-o DIR`, which exports a copy of the stored result files into `DIR`. These are
requests to the store and an export, and change no identity.

A prediction's identity is the run whose draws it uses and that run's problem,
the stored training data the filter conditions on, by stream, and every predict
option that changes the output, resolved to the values used: the mode
(`--mode`), `--horizon`, `--n-draws`, `--seed`, `--stream`,
`--scenario`/`--enable`/`--disable`, `--sweep`, `--by-chain`, the chains
`--exclude-chains`/`--chains` drop from the fit's cloud (so the two spellings of
one selection are one key), the particle count, the quantities and contrasts it
evaluates (those of the recorded model file, or of `--model`),
`--allow-forcing-extrapolation`, and, for a forecast, its target set (each
target's stream, period or instant, and design values where it reads any; D80,
D81), the design mode, and each `--covariate` statement, by forcing. A forecast
that reads no table forcing after its origin states none, and its identity has
no field for them. The build is in the ambient key, as for every result in the
run store. Two predictions that differ in the run, the build or any of those
inputs therefore never share a folder. The held-out outcomes are not in the
identity: they change no prediction (D79). The data are the problem's stored
training files, so an edit to the user's data file after the fit does not reach
a prediction. A forecast from a stored fit reads its targets from the problem's
`held_out/` files, computes no held-out score, and refuses `--realised-design`,
whose design values are held-out values; the held-out rows of a stored fit are
scored only by `camdl score` against outcome files (§2.2.4), and compared by
`camdl compare` (Appendix A, `camdl compare`). A rerun of `camdl predict` with
the same inputs is served from the store, without computing the prediction again
for every mode but a forecast, whose identity holds the target set it computes.
`predictive.json`, `observed.json` and `forecast.json` record the key under
`address_key` and the options under `options`. Every manifest of a prediction
(`predictive.json`, `observed.json`, the quantities manifest, `report.json`,
`forecast.json`) records the engine that made it as `run.json` does:
`engine_version`, the build fingerprint, and `engine_switches`, absent when no
switch was on. `fit table --quantity` reads the address its own free-forward
derivation writes. Inside the directory, `--exclude-chains` also names the files
(`predictive-excl3/`), so a chain-subset table says so in its name. A forecast's
files carry no such suffix (D86): `forecast.json`, `forecast-samples.tsv`,
`forecast/` and `report.json` keep their names, the directory alone is keyed by
the selection, and `forecast.json` states `options.excluded_chains` (`[]` for
the full cloud).

**Forcings read after the origin.** A forcing read from a data table is an input
at every time, and after a forecast's origin its values may have been fixed in
advance (a planned campaign, a scenario), observed later, or not known; camdl
cannot tell which. Before any filter runs, a forecast classifies each forcing
the model declares. A forcing is read after the origin when the run, up to the
end of the reach, evaluates its table at an argument after the origin. The
argument is `t − lag` for a forcing with a `lag`, which is never negative
(language spec §7, "Lagged forcing"): a positive lag is read after the origin
only once the reach exceeds the lag, and a lag that is a parameter may be
shorter than the reach, so it is taken to read every row of its table after the
origin. An `interpolated` forcing with `method = linear` (whatever its source)
is so read when its table has a row after the origin and the last argument
passes the origin; one with `method = constant` only when such a row lies at or
before the last argument. One with `method = spline` is so read whenever its
table has a row after the origin, whatever the lag and the reach, because a
natural cubic spline's value at every argument depends on every row. The
periodic kinds, a formula in `t`, a `piecewise` forcing (whose values the model
declares, and which holds its last value after its last breakpoint) and a
`bspline` are never so read; a forecast whose reach passes a `piecewise`
breakpoint says so in a note, on stderr and in `forecast.json`. A forecast that
reads a table forcing after its origin is refused until the user states what
those values are, with `--covariate FORCING=STATEMENT` (repeatable, one per
forcing):

```
error: forcing `pop` is read after the forecast origin and nothing says how.
  origin 1961-01-01, reach 8 weeks (to 1961-02-26); its table has rows to 1964-12-31.
  State one of:
    --covariate pop=prospective   the values after the origin were fixed or assumed at the origin
    --covariate pop=realised      they were observed after the origin
    --covariate pop=hold          carry the value at the origin forward
```

The times are dates under a model `origin`, and model time otherwise.

- `prospective` reads the table as it is: the forecast is conditional on those
  values. The word is the one a prospective design uses (`--design`).
- `realised` reads the table as it is, and labels the forecast "conditional on
  covariates observed after the origin": an ex-post evaluation, as a forecast
  under `--realised-design` is.
- `hold` holds the forcing at its table's value at the origin wherever its
  argument `t − lag` is after the origin. The held value is the table's
  interpolated value at the origin, so for a `linear` table it depends on the
  first row after the origin as well as the last row before it; for a `constant`
  table it is the last row's value at or before the origin. Every evaluation
  whose argument is at or before the origin is unchanged, bit for bit, so the
  filtering through the training rows is the same as without it. A held forcing
  is not evaluated past its table, so it raises no `forcing-outside-range`.
  `hold` is refused for a cubic-spline table (`method = spline`), whose value at
  every time depends on every row, so that cutting it at the origin would change
  the forcing before the origin too. A forcing with a `lag`, a parameter lag
  included, can be held: a model time at or before the origin reads the table at
  `t − lag`, at or before the origin, so the hold changes none of those values.

An indexed forcing (`pop[p in patch]`) is one forcing per level, and each is
stated by its underscore-joined name (`--covariate pop_p0=hold`), as `--sweep`
names an indexed parameter; the refusal names each level that needs a statement.

`--covariate` is refused outside a forecast, for a forcing the model does not
declare, for a forcing named twice, and for a forcing that needs no statement,
with the reason. A table that ends before the reach is still refused as
`forcing-outside-range` past its end, under `prospective` or `realised`, unless
`--allow-forcing-extrapolation` holds its last value; an estimated `bspline`
past its window is refused the same way. The statements are in the prediction's
identity (above) and in `forecast.json` (`covariates`, below).

The free-forward replay projects each row off its own recorded trajectories, so
it records a snapshot at every time a row reads — each data time and each
interval boundary — in addition to the model's output schedule. A data record
whose times fall between output times (data every 2.5 days against the default
1-day output) therefore needs no `output { trajectories { ... } }` line, which
would re-key the model and orphan the fit. The extra snapshots are internal to
the projection: they change no stored fit or run identity, and the quantity
tables stay on the declared output grid.

```
<run folder>/predict/<mode>-<12 hex>/predictive/<stream>.tsv
  scenario | sweep:<param>… | time | <dims…> | mode | horizon | treatment | role
           | quantity | fit_rhat_max | fit_ess_min
           | rhat_mean | ess_mean | rhat_pred | ess_pred
           | n_draws | q05 | q25 | q50 | q75 | q95

<run folder>/predict/<mode>-<12 hex>/observed/<stream>.tsv
  time | <dims…> | value
```

`predictive/<stream>.tsv` summarizes the model's distribution over the
observable as quantiles of sampled `y_rep` per
`(scenario, sweep point, time, stratum)`. Every axis is a column, so a new
predictive cell is more rows and never new consumer code. The `mode` column is
`free_forward` (run the fitted model forward from the start, `p(y_t | θ)`) or
`one_step` (`p(y_t | y_{1:t-1})`, re-running a bootstrap filter per posterior
draw and pooling over particles × draws); both stack under one header. The two
modes walk the step the fit ran at — the model's declared `dt` when it has one
(the one dt rule), else the fit's `[config] dt` — since either mode on another
step would replay a different process than the posterior describes. `horizon` is
a forecast target's position after its origin (as in `forecast/<stream>.tsv`)
and is empty on every row here, since neither mode has an origin. `quantity`
says what a band is of: `observation`, a draw `y` from the observation model, on
every row `camdl predict` writes; `expectation`, the observation model's mean
`E[y | x, θ]` with no observation noise drawn, is the other value, written by
`pfilter --save-filtered-streams`. Observed data are compared with an
`observation` band; an `expectation` band is narrower by the observation noise.

**Each row carries its role in the fit.** The `role` column is `trained` for a
row of the fit's training prefix of its stream (an `NA` row included),
`held_out` for a row of the bound file after that prefix
(`[data] holdout_after`), and `future` for a free-forward row past the last row
of the bound file: this prediction read no data row for this period. A
prediction from a fixed point (`camdl predict MODEL --params …`) given no
`--holdout-after` labels each of its data rows `unknown`, since nothing records
which rows the point was fitted to; with `--holdout-after` the rows it
conditions on are `trained`, as declared and not checked. A fixed point takes no
`[data.holdout]` file, so `--holdout-after` is its only way to declare its
training rows. (`target` is reserved for a forecast's targets, which carry a
horizon and a lead; these two modes write none.) Membership is the row's
position in its stream, read from the training partition the fit recorded and
checked against the data files; a fit that recorded none is refused. A
`one_step` band covers every row of the bound files. At a `held_out` row its
parameters were fitted without that row, and its state was filtered through
every earlier row of every stream this predict covers (all bound streams, or
those `--stream` names), earlier held-out rows among them, so it is not a
forecast from the end of the training rows (`--horizon` writes that one). For a
stream labelled by each period's start, the `one_step` rows' `time` is the
period's end while `free_forward` rows and `observed/` carry the label
(gh#1152), and the `free_forward` rows past the data are mis-timed (gh#1151);
`role` is read by row position and is right on both. `predictive.json` records
both sets:

- `parameter_set`: the fit's training partition, as `forecast.json` carries it
  (the `rule`, the training `extent` as `{t, date}`, and each stream's training
  `rows` and their digest).
- `state_sets`, keyed by `mode` value, one per section written. `free_forward`
  is `{"conditioning": "none", "streams": []}`: the band is never conditioned on
  a row. `one_step` is `"conditioning": "filtered"` with, per stream, the `rows`
  the filter walked, how many of them are `held_out_rows`, and the rows' digest.
  The filter walks a stored fit's training rows, so `held_out_rows` is 0; under
  `--stream` only the selected streams are listed.
- `holdout_files`, at the top level: `null` without `[data.holdout]`, and under
  it `{"read": false, "sources": […]}`. These two modes do not read the holdout
  files, so their rows are neither filtered through nor predicted as data rows.
  Every `one_step` row is then `trained`, and a `free_forward` row in a period a
  holdout file covers is `future`, not `held_out`. `holdout_after` and
  `[data.holdout]` describing the same split therefore do not yet give the same
  record (gh#1309).

**A downstream simulation inherits its fit's stochastic configuration (D78).**
Unless explicitly overridden, every downstream simulation inherits the
stochastic configuration recorded by its parent fit. For the binomial sampler
(`btpe` or `btrs`, the `binomial` key of the method), `camdl predict` reads the
sampler from the fit run's `run_inputs.json` (`binomial`), and every simulation
it runs draws with it: the one-step filters, the free-forward replay and its
`y_rep` draws, the forecast's filters and the contrast forks. `predictive.json`
and `forecast.json` record it as `binomial`. `camdl predict` has no flag that
overrides it. The guarantee is the same sampler and the same recorded
configuration, not the same random numbers: a predict does not reproduce the
draws the fit made. `simulate` does not follow this rule when it replays a fit
(`--draws posterior --fit FIT`, `--init-state fit`): it draws with BTPE whatever
the fit ran.

The `treatment` column is `posterior` when the band averages over a fit run's
draw cloud and `plug_in` when it pools copies of one fixed point. The label is
read off the cloud, which is built either from a fit's draws or from a point's
copies, so a posterior-labelled band over a single point is unrepresentable. A
`plug_in` row leaves `fit_rhat_max`, `fit_ess_min` and the per-row `rhat_*` and
`ess_*` cells empty. `fit_rhat_max`/ `fit_ess_min` are the producing leaf's
**rank-normalized split R̂** and **bulk-ESS** of Vehtari et al. (2021) — the
maximum R̂ and the minimum ESS over the leaf's estimated parameters — empty when
the leaf reported none, and `fit_ess_min` withheld entirely whenever any
assessed parameter has no pooled ESS rather than minimized over the ones that
do. Their names say whose numbers they are: the fit's worst parameter, repeated
identically on every row — provenance, not a statement about the value beside
them.

Beside them sit two per-row convergence channels, both reducing the same draws
the quantiles pool over but grouped by the chain each draw came from.
`rhat_mean`/`ess_mean` reduce the **latent expected value** `E[y | x_t, θ]`,
before observation noise: "do the chains agree about the expected trajectory
here?". `rhat_pred`/`ess_pred` reduce the **predictive draws**: "do the chains
give the same predictive distribution?". The two are not interchangeable. A
predictive draw carries observation noise, and that noise lands in the
within-chain variance, so where it is comparable to the between-chain
disagreement it swamps the numerator and `rhat_pred` is pulled toward 1 however
much the chains disagree — worst on overdispersed counts. Decide on `rhat_mean`;
read `rhat_pred` only for an interval genuinely dominated by irreducible
observation noise. An empty cell is a refusal (fewer than 2 chains, fewer than 4
draws per chain, a `draws.tsv` with no chain column, a constant row), never a
pass; the `one_step` mode leaves both pairs empty, because its cell pools over
filter particles as well as draws.

`camdl predict --by-chain` adds a leading `chain` column — `all` on the pooled
rows, the chain id on one additional band per chain, on **both** modes. The
pooled band is unchanged and remains the default object; without the flag no
`chain` column is written. A per-chain band carries no `rhat_*`/`ess_*` cell,
because those compare chains, and reports its own chain's `n_draws`.

The two modes answer different questions per chain, and a reader must know which
they are looking at. A **free-forward** per-chain band asks whether the chains
_project_ the same future; it runs free past the data, so a separation there
mixes disagreement about the fitted trajectory with extrapolation uncertainty. A
**one-step** per-chain band asks whether each chain _explains the observed
record_; it is re-anchored to the data at every step, so a separation there is
disagreement about the fitted trajectory itself, with the extrapolation removed.
The one-step half is therefore the sharper statement about mixing, and the
in-sample counterpart of the free-forward one.

Grouping the one-step pool by chain is a regrouping of the same reduction, not a
decomposition of the particle contribution: each chain's band pools its own
draws × particles exactly as the pooled band pools all of them, so the particle
pooling affects a per-chain band's width the same way it already affects the
pooled band's. That pooling is why one-step rows carry no `rhat_*`/`ess_*` at
all (an ESS from an autocorrelation over draws × particles is inflated by the
particle count — deferred, gh#798); it is not a reason to leave the rows pooled.

A chain whose one-step filter degenerated on some of its draws still bands, and
its `n_draws` reports what it actually pooled rather than what the subsample
gave it, so a narrower band there is legible as a thinner sample; the loss is
named on stderr. A chain that lost _every_ draw contributes no band at all and
is named too — banding an empty cell would publish `NaN` quantiles, which reads
as a band.

`--by-chain` adds no artifact address of its own, and composes with
`--exclude-chains`, which does. A `--by-chain` file is a strict superset of the
pooled one — its `all` rows are byte-identical, so re-running with the flag adds
rows and one column rather than replacing an artifact with a different object,
exactly as `--scenario` and `--sweep` do. A chain subset is a different
posterior, so it keeps its own keyed directory, and the two compose:
`camdl predict --by-chain --exclude-chains 3,5` writes
`predictive-excl3_5/<stream>.tsv` with a `chain` column. Chain ids are the fit's
own numbering and are never renumbered by an exclusion, so an excluded chain is
simply absent from the column and the ids that remain name the same chains as in
the pooled artifact.

`n_draws` is the cloud size behind each band. The `scenario` column is `fitted`
for the one-step rows, which are scenario- and sweep-agnostic by construction:
filtering _observed_ data through a counterfactual model is ill-defined. The
one-step filter runs the fitted model — the fit's own `scenario` / `enable` /
`disable` applied (§11.3) — so it fires exactly the interventions the fit's
likelihood fired.

`observed/<stream>.tsv` is the recorded value per `(time, stratum)` in the same
tidy keys. A hole — a scheduled but unobserved cell — renders as an empty
`value`, distinct from an observed zero. A consumer reads both files, joins on
`(time, <dims>)`, and plots `observed` over the `predictive` ribbon, one facet
per stratum, using `index_dims` from the `fit.meta.json` schema and no
run-store, DSL, or likelihood knowledge.

`predictive.json` (schema tag `camdl.predictive/v5`) records the `run_id` whose
draws it replays and declares, per stream, which columns are join coordinates
versus band versus per-cell diagnostics, the `value_kind`, and the band's
quantile levels; it also carries a `chain_selection` block when
`--exclude-chains` or `--chains` narrowed the cloud, so a chain-subset band is
never mistakable for a full-cloud one. The block is `excluded`, `kept`,
`n_total` and `summary`, a sentence naming the chains left out and the bias
toward the retained chains' mode; `forecast.json` carries the same block, from
the same function. Its `produced`, `skipped` and `failed` lists name the model's
observation streams (by leaf name) this artifact holds, leaves out and could not
produce; they follow the stream contract below.

**The tag is load-bearing, and a stored artifact must be read under its own.**
The versions do not carry the same numbers under the same column names:

| tag  | the two fit-provenance columns           | what they hold                                                                                                                                                                          |
| ---- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v1` | `rhat_max`, `ess_min`                    | classic Gelman–Rubin R̂ and a Geyer per-chain sum                                                                                                                                        |
| `v2` | `rhat_max`, `ess_min` — _the same names_ | rank-normalized split R̂ and bulk-ESS (gh#84), `ess_min` withheld rather than minimized when any parameter has none                                                                      |
| `v3` | `fit_rhat_max`, `fit_ess_min`            | the v2 statistics, renamed to say whose they are, plus the per-row `rhat_mean`/`ess_mean`/`rhat_pred`/`ess_pred` channels                                                               |
| `v4` | `fit_rhat_max`, `fit_ess_min`            | the v3 statistics; the `role` coordinate is added after `treatment`, and the manifest gains `parameter_set` and `state_sets`                                                            |
| `v5` | `fit_rhat_max`, `fit_ess_min`            | the v4 statistics; the mode-valued `horizon` column is renamed `mode`, `horizon` holds a forecast target's position (empty here), `quantity` follows `role`, and `role` gains `unknown` |

The v1 → v2 row is why the tag exists at all: the estimator changed and the
column names did not, so for those two versions the tag is the _only_ signal
that two artifacts in one store hold different quantities. Joining them without
keying on it compares a classic R̂ against a rank-normalized split R̂, which are
not the same number. The v3 rename is the first time that distinction is visible
in the column name itself. `observed.json` (`camdl.observed/v1`) is its sibling
for the observed series.

**`report.json` says what the run computed and what refused.** Schema tag
`camdl.predict-report/v2`, written on every `camdl predict`, with a `failures`
list that is empty when nothing refused and a `skipped` list that is empty when
every declared stream was covered:

```json
{
  "schema": "camdl.predict-report/v2",
  "failures": [
    {
      "kind": "evaluation_failed",
      "at": "free_forward",
      "reason": "scenario 'longer' declares …"
    }
  ],
  "skipped": [{ "stream": "weekly_cases", "reason": "no_bound_data" }]
}
```

**Every mode covers the same streams (D76).** A stream is _eligible_ when the
fit bound data to it. An eligible stream is produced. A stream the model
declares but the fit bound no data to, and that `--stream` did not name, is
_skipped_ with the reason `no_bound_data`: it was never eligible, so it is named
on stderr, in `report.json` and in the manifest, and does not change the exit
status. `--stream` (repeatable; a logical name selects all of its leaves)
narrows the run to the named streams, and a named stream that cannot be produced
is a failure. The exit status follows:

| situation                                                               | exit |
| ----------------------------------------------------------------------- | ---- |
| every eligible stream produced                                          | 0    |
| every eligible stream produced; streams with no data skipped            | 0    |
| a stream with no bound data named by `--stream`                         | 1    |
| several streams named by `--stream` and any one not produced            | 1    |
| no stream eligible (nothing is written)                                 | 1    |
| an eligible stream fails while it is produced (e.g. refused a forecast) | 1    |

The manifest — `predictive.json` for the free-forward and one-step modes,
`forecast.json` (`camdl.forecast/v5`) for a forecast — lists `produced` (leaf
names), `skipped` (`{"stream", "reason"}`) and `failed`
(`{"stream", "reason"}`), so a consumer reads completeness off the artifact
rather than the exit status. A forecast also skips an eligible stream none of
whose periods ends within the reach, with the reason `no_target_within_horizon`,
and fails it instead when `--stream` named it.

A forecast's tables are:

```
<run folder>/predict/forecast-<12 hex>/forecast/<stream>.tsv
  scenario | sweep:<param>… | <dims…> | t_start | t_stop | mode | horizon | lead
           | quantity | mean | q05 | q25 | q50 | q75 | q95 | n_samples

<run folder>/predict/forecast-<12 hex>/forecast-samples.tsv
  scenario | sweep:<param>… | sample | chain | draw | cloud_row | particle
           | stream | t_start | t_stop | horizon | quantity | value
```

`mode` is `forecast` on every row, `horizon` the target's position among its
stream's targets, `lead` its end minus the origin, and `quantity` is
`observation`: the bands and the samples are of draws from the observation
model.

`forecast.json` also records what made the forecast and what it holds:

- `engine_version`: the binary that wrote it, as `run.json` records it, beside
  the `run_id` whose draws it ran.
- `run_extent`: the model's `simulate { to }` (`model_end`), the end the run
  went to (`run_end`, the latest target end when that is later) and whether it
  was `extended` (D49). Interventions are planned to `run_end`; nothing else is
  extended.
- `samples`: what `forecast-samples.tsv` holds. Each posterior draw contributes
  `per_draw` joint samples (5), evenly strided over its `of_particles`
  particles; each sample is one particle's trajectory across every target. The
  bands and means in `forecast/` use every particle of every draw.
- Nothing about held-out rows is recorded, here or in the tables: the files hold
  only inputs to the predictions, so they are byte-identical when held-out rows
  change or are removed. The held-out rows (after the training prefix under
  `holdout_after`, or in a `[data.holdout]` file) select which targets have an
  observed value (D80); nothing filters them. The filter carries each value it
  can score as a held-out cell (D79), which its scoring likelihood evaluates per
  particle at the target's end and which enters no weight, resampling or state.
  A value the data checks refuse, and in a prospective forecast every value of a
  design-reading stream, is not scored, never refusing the prediction. The
  held-out rows' digests, which targets they measure and which row matched which
  target, and the log-likelihoods belong to the score artifact (forecast
  scoring, Stage 6).
- `targets`: each target's `stream`, `start` and `end`, `horizon` (its position
  among its stream's targets), `lead` (its end minus the origin), and `design`,
  the design values the prediction read (absent for a stream that reads none).
  `target_sources` names each stream's source: `declared_tiling`,
  `emit_schedule`, or `held_out_rows` for a window-column stream with no
  `emit_schedule`, whose horizons count its held-out rows and are not comparable
  with a tiled stream's; compare such targets by `lead`.
- `design` (D81): `null` when no target reads a design column; otherwise `mode`
  (`prospective`, from `--design` files; `ex_post`, from the held-out rows'
  realised values under `--realised-design`), `label` (an ex-post forecast is
  labelled a conditional score using realised future design, not an
  unconditional forecast), `streams`, and `files`: each `--design` file's
  `stream` and the `sha256` of its bytes (the path typed is not recorded).
- `covariates`: absent when the forecast reads no table forcing after its
  origin, passes no `piecewise` breakpoint, and has no table forcing whose
  values before the origin depend on rows after it. Otherwise `interpolation`
  (absent when empty), one `{forcing, method, cut, rows_after_cut}` entry per
  table forcing whose values before the origin depend on `rows_after_cut` rows
  after it, as a fit's `forcing_interpolation` records ("Table forcings across a
  cut", §2.2.1), whatever its statement; `forcings`, one entry per `--covariate`
  statement with the `forcing`, its `statement`, its `method`, its `table`
  (`source`, the `data = "path"` as written in the model or `null`;
  `rows_sha256`, the SHA-256 of its rows as the forecast reads them; the times
  of its `first` and `end` rows), its `lag` (`null`, a number, or `"parameter"`)
  and, under `hold`, `held` (`after`, the origin as a value of the table's
  argument, `on`, which says so, `"the table's argument, t − lag"`, and the
  `value` carried); `notes`, one per `piecewise` forcing whose breakpoints the
  reach passes; and, when any statement is `realised`, `label`:
  `"conditional on covariates observed after the origin"`.
- `chain_selection`, under a chain selection: the block `predictive.json`
  carries.
- `arms` (D50): the fitted model (`"baseline": true`), then each `--scenario`
  and `--sweep` point, in the order of the tables' rows, each with its
  `scenario`, its `sweep` values, `first_difference` (the earliest time it acts
  differently from the fitted model, at or after the origin, or `null` when
  nothing it changes acts within the run), and the `interventions_changed` and
  `parameters_changed`. `forecast/<stream>.tsv` and `forecast-samples.tsv` lead
  with a `scenario` column, and a `sweep:<param>` column per swept parameter
  (empty on the fitted model's rows). Sample `s` is the same draw and particle
  index in every arm. A per-sample difference across arms, `A[s] − B[s]`, is not
  a supported quantity: its spread is set by how the arms' random numbers are
  coupled (D83). Only the difference of means in `forecast-contrasts.tsv` is.
- `particles` (D85): `{"n", "source"}`, the filters' particle count per draw and
  where it came from: `flag` (`--particles`), `fit_pmmh` (the PMMH fit's own
  count) or `default` (500). `predictive.json` carries the same block when the
  one-step band ran. Whenever a filter runs, the output address keys the count
  `n` (as `options.particles`) and not its source, so `--particles` equal to the
  default addresses the same output as no flag.
- `filter_diagnostics` (D85): per draw, in the order of `draws`, the effective
  sample size of the filter's weights at its last assimilation at or before the
  origin, before resampling (`ess`), its time `t`, and `ess_over_n`. Every arm
  shares it. It characterises the Monte Carlo reliability of the state the
  forecast starts from; it is not a count of distinct ancestors and says nothing
  of the forecast's accuracy. The number of distinct parents at the final
  resampling is not recorded (`null`): the filter returns its resampling indices
  only when it records its full ancestry.
- `coupling` and `contrasts`, with more than one arm (D83). `coupling.method` is
  `shared_seed_per_particle_streams`: draw _m_ runs with the same seed in every
  arm and each particle with its own stream from it. `contrasts` names
  `forecast-contrasts.tsv` and defines its columns: `delta`, the arm's
  predictive mean minus the fitted model's, and `delta_mcse`, its Monte Carlo
  standard error. Per draw _m_, `d_m` is the arm's mean over its particles minus
  the fitted model's; per chain _c_ with `n_c` of the _M_ draws,
  `Var(Δ) = Σ_c (n_c / M)² · s²_c / ESS_c`, with `s²_c` the sample variance of
  the chain's `d_m` and `ESS_c` the effective sample size of a variant of
  Geyer's initial-positive-sequence estimator, over them in chain order. A chain
  whose `d_m` vary but carry fewer than `min_ess_per_chain` (10) effective draws
  withholds the error, with the reason in `mcse_note`. No quantile band of
  paired differences is published.

A _deterministic failure_ is a fact about the model as written that holds for
whatever produced it regardless of how often it occurs — an unresolvable
horizon, a scenario window this verb cannot honour, a `NaN`/±∞ where a number
was about to be banded. One occurrence is a defect, so the run exits **1**. What
it does not do is discard the objects that computed. Two kinds are recorded,
each at the site it refused:

- `evaluation_failed`, at `"free_forward"` — the tail as a whole refused. The
  one-step-ahead band is data-conditioned and never reaches the forecast
  horizon, so it cannot be invalidated by this: it is written, the tail's
  failure is recorded, and the run exits 1. `--mode one_step` remains the way to
  ask for that half alone — and because such a run never builds a tail, a
  scenario it could not have honoured is a plain usage error there, refused
  outright.
- `non_finite`, at `{"quantity": "<name>"}` with the `draw` that produced it —
  one `quantities {}` entry could not be banded. That entry writes no file and
  gets no manifest entry; every other quantity and every stream artifact is
  written, and the run exits 1. An entry that refused in one design cell is
  dropped from all of them, since the cells share one file and a partial file
  reads as a table over every cell.
- `evaluation_failed`, at `{"stream": "<leaf>"}` (free-forward and one-step) or
  `{"forecast_stream": "<leaf>"}` (forecast) — one stream could not be produced:
  `--stream` named it and the fit bound it no data, or the forecast refused it.
  Every other stream is written, and the run exits 1.

`simulate` renders the same quantity tables but has no report to write into, so
a non-finite quantity is an error there, named, with nothing written.

**A chain subset is a different address, not a rewrite of the pooled one.**
`camdl predict --exclude-chains 3,5` writes `predictive-excl3_5/<stream>.tsv` +
`predictive-excl3_5.json`, and likewise `contrasts-excl3_5/`,
`quantities-model-excl3_5/` with its manifest `quantities-excl3_5.json`, and
`report-excl3_5.json`; the full cloud keeps the bare names. The key is the
excluded **set**, so `3,5` and `5,3` are one address. Without this a subset
would silently replace the run's canonical predictive and a second subset would
replace that, leaving only a stamp inside the file that had already been
overwritten. `observed/` is not keyed — it is the recorded data, identical under
any selection. A prediction is a result of its own: `camdl show <prediction>`
(its id prefix or folder, which `camdl predict` prints) lists its files, and
`camdl cat <prediction> predictive/<stream>` reads one. `camdl show` and
`camdl cat` on the fit run list and read only the run's own files. `--chains`
keeps the named chains instead of dropping them, and is addressed by the chains
it drops: on a six-chain fit `--chains 0,1,2,4` and `--exclude-chains 3,5` both
write `predictive-excl3_5/`. A `--chains` list that names every chain of the fit
selects nothing and is refused.

**Calendar semantics travel with the artifact.** `predictive.json`,
`observed.json`, `quantities.json`, the per-chain `trajectories.json`, every
leaf's `run.json` (§9.5) and `fit.meta.json` each carry a top-level `calendar`
block, written through one type (`ir::caltime::CalendarAxis`) so every artifact
agrees:

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
diff-stable axis; `calendar` is additive metadata, not a re-encoding of it. A
date camdl renders from the axis (`--dates`, the prequential `date` column)
loads back through `--data` to the time it was rendered from (`docs/dates.md`,
"Getting dates back out").

#### 2.2.3 Which object a quantity is read on

`camdl predict` also writes the model's `quantities {}` block to
`quantities-model/<name>.tsv` plus a `quantities.json` manifest. The run store
keeps every result kind's name, `quantities` among them, out of a result's
top-level directories; `simulate --quantities-out DIR` writes `DIR/quantities/`.
Those numbers are not all folded over the same object, and the manifest says
which.

Every banded row there carries `rhat` and `ess` after `n_draws`: the
rank-normalized split R̂ and bulk-ESS of _that row's_ reported value, reduced
over the same draws the band pools but grouped by the chain each draw came from.
One pair, not two, because a quantity has a single value per draw. These are the
reported estimands, so they are the first convergence numbers to read — and they
routinely disagree with the parameter R̂ in either direction: a reportable
quantity is often far better determined than the parameters behind it. What the
value carries decides which kind of R̂ it is: a `state` or `derived` quantity is
a deterministic function of the trajectory and the parameters, so its R̂ is
undiluted; a quantity whose manifest `source` is `observations` reduces sampled
`y_rep`, carries observation noise, and so reads like `rhat_pred`.
`simulate --quantities-out` has no chains behind it and writes neither column.

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

#### 2.2.4 `camdl score` score artifact

`camdl score` scores a stored prediction against the observed outcomes of the
rows it predicted. The outcomes change the score and never the prediction: a
forecast made today and scored next month is one stored prediction with one
score.

```bash
camdl predict @sir --mode forecast --horizon "8 weeks" --label sir_fc
camdl score @sir_fc --data cases=london.tsv
camdl predict measles.camdl --params he2010.toml --data cases=london.tsv \
    --mode one_step --holdout-after 1960-12-31 --label he
camdl score @he --data cases=london.tsv
```

The positional argument is the prediction's handle (`@label`, a run-id prefix or
its folder). `--data STREAM=FILE` gives one outcome file per stream source: for
a stored fit, the file the fit was given, held-out rows included; for a fixed
point, the file the prediction was given, extended past `--holdout-after`.
`--data-held-out STREAM=FILE` gives instead a file of a stream's held-out rows
alone, with the columns of its data file: every row must end after the last row
the prediction conditioned on for that stream, which `score` reads from the
store, or the file is refused, naming the first row that does not. The rows are
read as the whole file they complete (the stored conditioning rows, then these),
so one outcome stated by `--data` or by `--data-held-out` is one score; a stream
named by both is refused. The flag suits any prediction, and is the natural
spelling for a fit whose held-out rows are a `[data.holdout]` file. `score`
takes no flag for the prediction: its parameter source, mode, reach, particles,
seed and scenarios are in its identity. `--store`, `--label`, `--force` and
`-o DIR` mean what they mean for `camdl predict`.

**The outcome files must agree with the prediction.** Each file is read by the
loader the prediction read its data with, and every row the prediction
conditioned on must be in it, equal by row digest, before the rows after them. A
file that lacks one or differs in one is refused, naming the stream, the row and
its value: its held-out rows would belong to another series. A stream with no
`--data` file is scored on the rows the prediction conditioned on only.

**The score computes the prediction again.** The stored files keep summaries,
and a forecast keeps 5 particles per draw, so the score runs the prediction's
own computation with each draw's own seed and the outcomes bound. A bound
outcome is a held-out value that no filter weight reads and that draws no random
number, so the prediction computed has the stored prediction's run id and writes
its files byte for byte; a difference is refused as an internal error and no
score is written.

What is scored, by mode:

- **Free-forward.** Every bound row the prediction emitted a band for: each
  draw's observation density of the outcome at that draw's simulated state. On
  the ODE backend this is the exact density along each draw's trajectory. An
  outcome row past the prediction's data that no emitted row measures (a stream
  whose likelihood reads a data column stops at its data) is not scored, and is
  named in `score.json`'s notes.
- **One-step** (chain-binomial). The filter runs on past the prediction's last
  conditioning row through the outcome rows after it, predicting each row from
  the particles before it and then assimilating it. Its bands for those rows are
  the score's and are written to no prediction file. A particle the filter
  killed has density zero in every stream, as in the joint, and its draw is no
  sample. A draw whose filter degenerates on a row after the conditioning rows
  has density zero at every particle there, and its pass stops: it scores no
  later row, and `score.json`'s notes name the draw and each row it leaves
  unscored. A row's score is then the log of the mean density over the particles
  of every draw that reached it, finite while some draw gave the outcome
  positive density and `-inf` when none did; its other rules use the samples of
  the draws that reached it alive. A row no draw reached is counted in
  `summary.tsv`'s `n_unscored` and written to no other file.
- **Forecast.** Each target whose period an outcome row measures: the
  observation density at each particle at the target's end, recorded before
  anything could be assimilated there.

**Rules.** For one row, with `S = M·N` samples (`M` draws, `N` particles per
draw; `N = 1` for free-forward):

- `log_score`: `logsumexp_s ℓ_s − log S`, `ℓ_s` the log density of the outcome
  at sample `s`'s state: the log of the mean density, never the mean of per-draw
  logs. At a fixed point with `--n-draws 1` under one-step, each row's score is
  the log of the filter's incremental likelihood at that row (pomp's
  `cond_logLik`, King, Nguyen & Ionides 2016), and their sum is `pfilter`'s
  log-likelihood with the same particle count and the seed the prediction ran
  that copy's filter with.
- `crps`: the fair sample estimator (Ferro 2014).
- `wis`, `wis_dispersion`, `wis_over`, `wis_under`: the weighted interval score
  of Bracher et al. (2021) and its parts, from the samples' quantiles at the 23
  levels of the US COVID-19 Forecast Hub (0.01, 0.025, 0.05, 0.1, …, 0.9, 0.95,
  0.975, 0.99).
- `pit`: the randomized probability integral transform, its uniform draws from
  the prediction's seed.
- `in_50`, `in_90`: 1 when the outcome lies in the central 50% or 90% interval
  of the samples, ends included.

A score that is `-inf` is written `-inf`, and so is a sum that includes one.
When a row has one sample (a free-forward prediction at a fixed point with
`--n-draws 1`), its CRPS, WIS, PIT and coverage come from that single sample,
and `score.json`'s notes say so.

Under one-step and forecast prediction a row with `stream` `joint` holds the log
score of every stream observed at one time together, from each sample's summed
log densities; a forecast also has one `joint` row over every scored target,
with `lead` empty. Only the log score is computed for a `joint` row.

**Files.** The score is stored inside the prediction, at
`scores/score-<12 hex>/`:

- `scored.tsv`: one row per scored row or target, with the columns
  `scenario [sweep:<p>] stream source <dims> t_start t_stop [date_start date_stop] mode origin horizon lead role parameters quantity mean q05 q25 q50 q75 q95 observed log_score crps wis wis_dispersion wis_over wis_under pit in_50 in_90`.
  `stream` is the observation block, and `source` the data source it reads: two
  blocks reading one source (`weekly from surv`, `recovered from surv`) are two
  streams. `role` is `trained`, `held_out`, `future` (every forecast target) or
  `unknown` (a fixed point with no `--holdout-after`). `lead` is a forecast
  target's time from the origin, or a one-step row's period (for an instant row,
  the time since its stream's previous row); it is empty for free-forward rows.
- `summary.tsv`: one row per (scenario, sweep point, stream, stratum, mode,
  role, lead), with
  `n n_unscored log_score_sum log_score_mean crps_mean
  wis_mean coverage_50 coverage_90`.
  Rows of different streams, modes, roles or leads are never pooled.
- `score.json`: the prediction's run id; `parameters` (`posterior`; `winner`, an
  optimizer run's winning point, whose conditioning is verified from the run's
  training rows; or `fixed_point`, a point given by a file); `conditioning`,
  what the parameters were fitted to (`hold_out_tail`, from the end of the fit's
  last training row, for a fit with held-out rows or a score of outcome rows
  past the fit's data; `in_sample` otherwise; `hold_out_declared` for a fixed
  point's `--holdout-after`, `unknown` without one); per mode the provenance,
  the basis (`filtered`, `ode_free_forward`, `simulated_free_forward`) and the
  draws and samples per draw; the reach, the particle count and the seed; the
  rules' settings; the forecast's covariate statements; `labels`
  (`conditional on covariates observed after the origin` under
  `--covariate NAME=realised`, `held out as declared, not verified` for a fixed
  point's held-out rows); per stream and mode the rows scored and their digest;
  the sets of streams the `joint` rows are of; the notes; and `forcings_held`,
  each forcing the computation held outside its range under
  `--allow-forcing-extrapolation`, which `score` also prints on stderr.

The identity of a score is the prediction, each stream's outcome lines after the
prediction's conditioning rows (stored as a data table), and the interval levels
and PIT seed. Editing a held-out outcome makes another score of the same
prediction. `camdl show` and `camdl cat` read a score; `cat` prints its
`summary.tsv` by default.

References: Bracher, Ray, Gneiting & Reich (2021), _PLOS Comput. Biol._ 17(2),
e1008618; Ferro (2014), _Q. J. R. Meteorol. Soc._ 140(683), 1917–1923; King,
Nguyen & Ionides (2016), _J. Stat. Softw._ 69(12).

### 2.3 Sweeping a fit over a fixed parameter

`camdl fit run <config> --sweep NAME=SPEC` applies each sweep point to the
resolved `[fixed]` block before identity is computed. A sweep point is therefore
a different problem, and each point gets **its own problem folder**, a sibling
with the same label:

```
<build>/fit/<model stem>-<12 hex for rho = 0.5>/runs/if2-<12 hex>/seed_1-<12 hex>/
<build>/fit/<model stem>-<12 hex for rho = 0.7>/runs/if2-<12 hex>/seed_1-<12 hex>/
```

There is no nesting of sweep points under a shared folder: within a problem, the
layout of a swept fit is identical to that of an unswept one.

### 2.4 Simulation Result Layout

`camdl simulate` stores each cell, one realisation at one parameter point, one
scenario and one seed, as one `sim` result in the run store (§4.4). The layout
is that of `docs/dev/proposals/2026-11-20-run-store-api.md` ("Layouts, kind by
kind"):

```
<root>/<version>-<12 hex of the ambient key>/          # the build, written <build>
  compiled_model/<model stem>-<12 hex>/                 # the compiled model, stored once
  sim/<model stem>-<12 hex>/<backend>-dt<dt>-<12 hex>/<params label>-<12 hex>/<scenario>-<12 hex>/seed_<seed>[-r<n>]-<12 hex>/
    identity.json
    run.json
    traj.tsv
    reactive_log.tsv                                    # when a reactive policy was active
```

A cell's identity is the compiled model it depends on, its settings (backend,
`dt`, the cell's start and end, a restored `--init-state` row,
`allow_degenerate_rates`, `allow_forcing_extrapolation`, `--no-flows` and the
`--columns` allow-list), every parameter's value after all sources, the
interventions its scenario keeps, and its process seed. Each folder name is a
label followed by 12 hex digits of that group's key; the label is not part of
the identity. The params label is `base`, or a sweep point's or draw row's
`name=value` pairs joined by `_`, or `draws` when that text exceeds 96 bytes.
The scenario label is the scenario's name, or for the ad-hoc arm of a run
without `--scenario` the label its flags derive (§5.5): `(none)` with no
`--enable`, `--disable` or `--param`, which a folder name writes `_none_`, and
`(none)+vaccination` under `--enable vaccination`, written `_none__vaccination`
(every character outside `a-z`, `0-9`, `_`, `-` and `.` becomes `_`). The seed
label is the seed typed: `--seed 7 --replicates 3` stores `seed_7`, `seed_7-r1`
and `seed_7-r2`, and each value of `--seeds` is its own `seed_<n>`.

Observations, the event log and generated quantities are by-products stored
inside the cell folder, each with its own `identity.json` and `run.json`; the
cell's own record is not changed by them (§4.4). No combined table is stored:
`-o`, `--stdout` and `camdl cat @LABEL` render it from the stored cells.

### 2.5 Per-result metadata

Every result folder in the run store holds an `identity.json` and a `run.json`,
the run store's records, whose format is that of
`docs/dev/proposals/2026-11-20-run-store-api.md`. Neither a `sim` cell's nor a
fit run's `run.json` carries an `output_schema`.

### 2.8 Profile Result Layout

`camdl profile` stores each point of its grid in the run store (§4.4) as one
result of kind `profile_point`: one start of one cell at one seed, a fit of the
problem with the focal parameters held at the cell's values, written in place
while it works. The layout is that of
`docs/dev/proposals/2026-11-20-run-store-api.md` ("A profile, and a grid that
grows"):

```
<root>/<build>/
  compiled_model/<model stem>-<12 hex>/          # the compiled model, stored once
  data_table/<stream>-<12 hex>/                  # each bound data table, stored once
  profile_point/<model stem>-<12 hex>/           # one group per problem
    <name>=<value>__…-<12 hex>/                  # the focal values, to four decimals
      <algorithm>-<12 hex>/                      # the method and its settings
        seed_<n>-start_<i>-<12 hex>/             # one point
          identity.json  run.json
          mle.toml                               # final_loglik, [focal], [mle], [diagnostics]
          provenance.json                        # where each value came from; wall time
```

The problem is a fit problem (§2.2) in the same store. `--fit @RUN` names a
stored fit run, and its problem is the profile's. `--fit FIT.toml` names a fit
file, whose problem is resolved and stored as `camdl fit run` stores it, so a
profile and a fit of one file share one stored problem. The store is found as
§2.1 states, so the problem is stored where `fit run` of the same file stores
it. `--fixed`, `--fixed-file`, `--table`, the scenario flags, the two policies
and `--dt` are written into the problem that is stored: `--fixed NAME=VALUE`
holds `NAME` and takes it out of `[estimate]`, a conditional problem of its own,
and so does each name a `--fixed-file` gives other than a swept one. Each value
they replace, and each parameter they take out of `[estimate]`, is reported at
info and recorded in every point's `provenance.json` (§1.3). The cells compute
from the stored problem: its compile, its training rows and its values. A
profile without `--fit`, and `--data` or `--obs` beside it, are refused
(gh#1354). A `--starts from_params` value for a parameter the problem neither
estimates nor fixes is refused, naming it.

A point's identity holds the problem, the focal values, the method (the
algorithm with its settings, `dt`, each estimated parameter's domain and IF2
step, the simplex groups, under IF2 the `--pf-max-substeps` budget and the
binomial sampler, and for `nl-lbfgs` the compile with the state Jacobian), the
start's values and a seed. The compile with the state Jacobian is of the model
file the profile is given, where it lies, so the files the model reads by a
relative path are found; it is rewritten from the problem's data and refused
unless it is the problem's model. It holds nothing about the grid: a larger grid
finds every point a smaller one stored and computes only the others. The seed is
derived from `--seed` (or one of `--seeds`), the focal values and the start's
values, so a point has one seed in every grid. Starts are drawn once for the
invocation, so the start's index is a label. Starts with equal values (the
`from_params` and `from_mle` rules start every chain at one point) are
independent runs: the second and later of them take a replicate index, the
number of earlier starts with the same values, into the seed. A start with no
equal earlier start takes no index. So `--n-starts 4` after `--n-starts 3`
computes only the fourth start.

A point whose log-likelihood is not finite is a completed point with
`final_loglik = -inf`: the failure follows from its identity. The profile table
(§10.12) is rendered from the points the invocation asked for, and is not
stored. `--force` computes every point again, moving the stored ones to the
quarantine. `--label L` names the invocation; `@L` is found again through its
record.

### 2.9 The result-file contract

The files each kind of result stores, which a program reading the store relies
on. A kind's files are fixed by the kind and its inputs; the folders they sit in
are not part of the contract (§2.1). Three content artifacts each store one
file: `compiled_model` its `model.ir.json`, `quantities` its `quantities.json`
and `data_table` its `table.tsv`, beside the provenance files their writer
records. The fourth, `model_source`, stores the model file and its views, listed
below.

<!-- The result-file contract below is generated from the file enums of rust/crates/cli/src/result_files; run `make update-result-file-contract` after changing them. -->

Each kind's files, by logical name: the stored name, the format, when a result
holds the file, what it holds and, for a table, its header row. A provenance
file is recorded with its digest and not compared when a result is computed
again. `chain_N` stands for each chain, numbered from 0. A run that holds any
other file, or lacks one it declared, is not stored.

Each kind's contract version, which an output manifest records with each result
(§2.10), is given with its files; it is bumped when a file is added, removed or
renamed, or a column changes. Each of the three single-file content artifacts
has a contract version of its own: `compiled_model` 1, `quantities` 1 and
`data_table` 1. The logical name of each one's file is its stored name up to the
first `.`: `model`, `quantities` and `table`.

#### `fit_run`

One fit run of a problem. Its files depend on its method and on how each chain
ended, as `chain_starts.tsv` records it: `accepted` (ran to the end), `skipped`
(ran and was dropped, IF2) or `refused` (never ran). Contract version 1.

- `method`: `method.toml`, TOML (every method). The run's `[method]` table as
  merged, with its resolved step tables.
- `summary`: `summary.json`, JSON (every method). `camdl.fit-run-summary/v1`:
  the run's fit summary, without times, paths or a baseline.
- `inputs/starts`: `inputs/starts.<ext>`, TSV (a starts file named directly). A
  starts file named directly, as read.
- `draws`: `draws.tsv`, TSV (mh, pmmh, pgas). The thinned post-warm-up draws of
  every chain, with every parameter, estimated and fixed. Header: `chain`,
  `draw`, every parameter, estimated and fixed.
- `chain_N/trace`: `chain_N/trace.tsv`, TSV (mh, pmmh, each chain; a refused
  chain's holds no row). One chain's every step: the log densities, whether it
  was accepted, and the parameters. Header: `step`, `log_likelihood`,
  `log_posterior`, `accepted`, each estimated parameter.
- `chain_N/trace`: `chain_N/trace.tsv`, TSV (pgas, each chain; a refused chain's
  holds no row). One chain's every sweep: the log densities, the sampler's
  diagnostics and the parameters. Header: `sweep`, `log_complete_data_ll`,
  `log_posterior`, the sampler's per-sweep diagnostics, then each estimated
  parameter.
- `chain_N/resume_state`: `chain_N/resume_state.bin`, binary (mh, pmmh, pgas,
  each chain that ran). One chain's internal state for `--resume`.
- `sampler_summary`: `mh_summary.json`, JSON (mh; pmmh and pgas as
  `pmmh_summary.json` and `pgas_summary.json`). The sampler's summary: the
  convergence and acceptance figures.
- `chain_starts`: `chain_starts.tsv`, TSV (if2, mh, pmmh, pgas). Each chain's
  starting point, and every start a chain tried and was refused. Header after
  `#` comment lines: `chain_id`, `attempt`, `status`, `source`, each estimated
  parameter, `ess`, `reason`.
- `diagnostics`: `diagnostics.json`, JSON (every method but pfilter). An array
  of findings, each with `kind`, `severity`, `message` and `method`.
- `fit_state`: `fit_state.toml`, TOML (every method but pfilter). The run's
  state: its method, chains and verdict.
- `chain_N/trajectories`: `chain_N/trajectories.tsv`, TSV (pgas, each chain that
  ran to the end and saved paths (`n_trajectories` above zero,
  `trajectories.n_saved` above zero in the summary)). One chain's saved
  posterior paths, stacked, one row per draw and time. Header after `#` comment
  lines: `chain`, `draw`, `time`, the step's bounds or date, each compartment,
  each `flow_<name>` and each projection.
- `chain_N/trajectories_meta`: `chain_N/trajectories.json`, JSON (as
  `chain_N/trajectories`). What one chain's `trajectories.tsv` holds: its draws,
  columns and granularity.
- `cross_chain_compat`: `cross_chain_compat.json`, JSON (pgas, when two or more
  chains ran to the end). The log density of each chain's final path under every
  other chain's parameters.
- `filter_ess`: `filter_ess.tsv`, TSV (pgas, when its summary holds the
  `filter_ess` block). The conditional filter's effective sample size at every
  observation, pooled over sweeps. Header: `chain`, `obs`, `time`, `ess_mean`,
  `ess_min`, `n_sweeps`.
- `latent_convergence`: `latent_convergence.tsv`, TSV (pgas, when its summary
  holds the `latent_convergence` block). Whether the chains agree about the
  latent states, per cell of the paths.
- `chain_N/parameter_traces`: `chain_N/parameter_traces.tsv`, TSV (if2, each
  chain: IF2 runs a chain whose every start was refused from its last one). One
  chain's estimate at every iteration, with its log-likelihoods. Header after
  `#` comment lines: `iteration`, `loglik`, `if2_perturbed_loglik`, each
  estimated parameter.
- `chain_N/final_params`: `chain_N/final_params.toml`, TOML (if2, each chain
  that produced a result, as its row in `chain_evaluations.tsv` records). One
  chain's final estimate.
- `final_params`: `final_params.toml`, TOML (if2). The winning chain's estimate:
  the same values as `mle_params`.
- `mle_params`: `mle_params.toml`, TOML (if2, nl-sbplx, nl-bobyqa, nl-lbfgs).
  The point estimate, parameters at the top level and a `[provenance]` table;
  read by `--params`.
- `chain_evaluations`: `chain_evaluations.tsv`, TSV (if2). Each chain's final
  estimate, evaluated by a fresh filter, with its standard error and ESS. Header
  after `#` comment lines: `chain`, `loglik`, `se`, `ess_mean`, `ess_min`,
  `ess_min_step`, `n_neg_inf_incr`, each estimated parameter.
- `iteration_diagnostics`: `diagnostics.tsv`, TSV (if2). Each chain's
  log-likelihoods at every iteration. Header after `#` comment lines: `chain`,
  `iteration`, `loglik`, `if2_perturbed_loglik`.
- `chain_results`: `chain_results.tsv`, TSV (nl-sbplx, nl-bobyqa, nl-lbfgs).
  Each search's end: its log-likelihood, status, evaluations and estimate.
  Header: `chain`, `loglik`, `status`, `n_evals`, each estimated parameter.
- `logliks`: `logliks.tsv`, TSV (pfilter). Each replicate filter's
  log-likelihood. Header: `replicate`, `loglik`.
- `prequential_joint`: `prequential.tsv`, TSV (pfilter, when
  `record_prequential` is on). At each observation time, the joint
  one-step-ahead log score and the ESS. Header: `t`, `log_score`, `ess`.
- `prequential_streams`: `prequential.streams.tsv`, TSV (pfilter, when
  `record_prequential` is on). At each observation time and stream, the
  predictive quantiles and the scores. Header: `t`, `stream`, `y_obs`,
  `y_pred_q05`, `y_pred_q25`, `y_pred_q50`, `y_pred_q75`, `y_pred_q95`,
  `log_score`, `crps`, `pit`.
- `prequential_record`: `prequential.json`, JSON (pfilter, when
  `record_prequential` is on). The prequential record: the scores and each
  stream's predictive samples.
- `fit_config`: `fit.toml`, TOML, provenance (every method). The fit
  configuration file as given.
- `run_inputs`: `run_inputs.json`, JSON, provenance (every method). What the run
  was asked to do and its best log-likelihood and chain.
- `fit_meta`: `fit.meta.json`, JSON, provenance (every method). The run's
  sidecar: the data digests, the roles and the training partition.
- `execution`: `execution.json`, JSON, provenance (every method). The attempt's
  wall times, process and cost.
- `opt_trace`: `opt_trace.tsv`, TSV, provenance (nl-sbplx, nl-bobyqa, nl-lbfgs).
  Each search's every evaluation and its running best. Header: `chain`, `eval`,
  `loglik`, `best_loglik`, each estimated parameter.

#### `model_source`

The model file one command compiled, with its name, what the compiler shows of
it, and the quantities its compile produced, on which it depends. No simulation
or fit depends on it; the compiled model's listing in each invocation record
names, as its `source`, the model source of the file that invocation compiled. A
command that stores a result from a stored one (`predict RUN`, `score`,
`profile MODEL --fit RUN`) lists the stored compiled model in its own record
with the source of the file it was given or, when it was given none, the one the
stored result's invocation recorded, and lists that source too. Contract
version 1.

- `source`: `source.camdl`, camdl (always; `source.ir.json` for a compiled
  model). The model file as written.
- `docs`: `docs.json`, JSON (always). The model's `description` and its `#'`
  documentation, by declaration.
- `declared`: `declared.json`, JSON (always). The model as its author declared
  it.
- `render`: `model.render.json`, JSON (a `.camdl` model that camdlc rendered).
  The model's mathematics as KaTeX strings, as `camdlc render --format json`
  writes them.
- `graph`: `model.graph.json`, JSON (a `.camdl` model that camdlc rendered). The
  compartment flow graph, as `camdlc render --format graph` writes it.

#### `fit`

A fit problem: the model, the data and the rows that train. Contract version 1.

- `problem`: `problem.toml`, TOML (always). The problem in fit-file syntax: the
  model, the data, the estimated and fixed parameters.
- `partition`: `partition.json`, JSON (always). Which rows of each stream train,
  and where training ends.
- `inputs/<source>`: `inputs/<source>.<ext>`, TSV (each stream source). One
  stream source's training lines, as read.
- `inputs/tables/<name>`: `inputs/tables/<name>.<ext>`, TSV (each external
  table). One external table, as read.
- `held_out/<stream>`: `held_out/<stream>.tsv`, TSV (each stream with held-out
  rows). One stream's held-out rows, without their values.
- `synthetic/truth`: `synthetic/truth.toml`, TOML (a problem on a generated
  dataset). The true parameters the dataset was generated at.
- `synthetic/truth`: `synthetic/truth.toml`, TOML, provenance (a real-data fit
  with a `[synthetic]` table). The true parameters `fit recovery` compares
  estimates with.
- `synthetic`: `synthetic.toml`, TOML, provenance (a real-data fit with a
  `[synthetic]` table). The fit file's `[synthetic]` table.
- `model`: `model.camdl`, camdl, provenance (always; `model.ir.json` for a
  compiled model). A model file that compiles to the problem's model: the one
  the command that first stored the problem was given. Each invocation's own
  model file is its model source's `source`.

#### `prediction`

A prediction from a fit run's posterior or from a fixed point. A name carries a
suffix when chains were excluded. Contract version 2.

- `predictive`: `predictive.json`, JSON (always). A part's manifest: its schema,
  its tables and how they were made.
- `predictive/<stream>`: `predictive/<stream>.tsv`, TSV (each stream predicted).
  One stream's predictive or forecast bands, per time, mode and quantity.
- `observed`: `observed.json`, JSON (when observations are bound). A part's
  manifest: its schema, its tables and how they were made.
- `observed/<stream>`: `observed/<stream>.tsv`, TSV (each stream observed). One
  stream's observed values: `time`, a column per dimension of the stream, and
  `value`.
- `forecast`: `forecast.json`, JSON (a forecast). A part's manifest: its schema,
  its tables and how they were made.
- `forecast/<stream>`: `forecast/<stream>.tsv`, TSV (a forecast, each stream).
  One stream's predictive or forecast bands, per time, mode and quantity.
- `forecast_samples`: `forecast-samples.tsv`, TSV (a forecast). One row per
  forecast sample: its draw, particle, stream, time and value.
- `forecast_contrasts`: `forecast-contrasts.tsv`, TSV (a forecast with
  contrasts). Each forecast contrast's samples.
- `quantities`: `quantities.json`, JSON (when the model has quantities). A
  part's manifest: its schema, its tables and how they were made.
- `quantities-model/<quantity>`: `quantities-model/<quantity>.tsv`, TSV (each
  quantity). One quantity's bands, per time.
- `contrasts`: `contrasts.json`, JSON (a fit run's prediction when the model has
  contrasts). A part's manifest: its schema, its tables and how they were made.
- `contrasts/<contrast>`: `contrasts/<contrast>.tsv`, TSV (each contrast
  forked). One contrast's bands, per time.
- `report`: `report.json`, JSON (always). What failed while the prediction was
  made, and which streams were skipped.

#### `score`

A stored prediction scored against its outcomes. Contract version 1.

- `scored`: `scored.tsv`, TSV (always). One row per scored row or target.
- `summary`: `summary.tsv`, TSV (always). One row per stream, mode, role and
  lead.
- `score`: `score.json`, JSON (always). The parameter source, the basis, the
  conditioning and the notes.

#### `pfilter`

A particle filter run at one parameter point. Contract version 1.

- `loglik`: `loglik.toml`, TOML (always). `loglik`, `n_replicates` and
  `n_particles`: the log-likelihood estimate.
- `replicates`: `replicates.tsv`, TSV (with `--replicates` above 1). Each
  replicate filter's log-likelihood.
- `provenance`: `provenance.json`, JSON, provenance (always). The parameter
  values as resolved, with where each came from.

#### `trace`

`--trace`, a by-product of a filter run. Contract version 1.

- `trace`: `trace.tsv`, TSV (always). Per observation, the log-likelihood
  increment, the ESS and the one-step predictive quantiles.

#### `pf_health`

`--pf-health`, a by-product of a filter run. Contract version 1.

- `pf_health`: `pf_health.tsv`, TSV (always). Per observation: `time`, `ESS`,
  `ESS_frac`, `tau2`.

#### `filtering`

`--save-filtering`, a by-product of a filter run. Contract version 1.

- `filtering`: `filtering.tsv`, TSV (always). Every particle's state and
  log-weight at every step.

#### `filtered_streams`

`--save-filtered-streams`, a by-product of a filter run. Contract version 1.

- `filtered_streams`: `filtered_streams.tsv`, TSV (always). Each stream's
  conditional mean, summarised over the swarm before and after each step is
  scored.

#### `final_state`

`--save-final-state`, a by-product of a filter run. Contract version 1.

- `final_state`: `final_state.tsv`, TSV (always). The swarm after the last
  observation.

#### `saved_paths`

`--save-paths`, a by-product of a filter run. Contract version 2.

- `paths`: `paths.tsv`, TSV (always). Sample paths from the smoothing
  distribution, traced through the ancestry: at each filter stop the
  compartments, each transition's flow and each overdispersed transition's rate
  noise over the interval since the previous stop, and each stream's projection.

#### `prequential`

`--save-prequential`, a by-product of a filter run. Contract version 1.

- `joint`: `prequential.tsv`, TSV (always). At each observation time, the joint
  one-step-ahead log score and the ESS.
- `streams`: `prequential.streams.tsv`, TSV (always). At each observation time
  and stream, the predictive quantiles and the scores.
- `record`: `prequential.json`, JSON (always). The prequential record: the
  scores and, unless `--no-save-samples`, each stream's predictive samples.

#### `sim`

One simulated cell. Contract version 2.

- `traj`: `traj.tsv`, TSV (when the engine ran). The trajectory at the output
  times: `time`, `t_start`, `t_stop`, each compartment and each `flow_<name>`.
- `reactive_log`: `reactive_log.tsv`, TSV (when the run kept a reactive policy's
  firings). Each firing of a reactive policy.
- `failure`: `failure.toml`, TOML (in place of `traj` when the engine stopped
  with an error). The operation and the error the engine stopped with.
- `provenance`: `provenance.json`, JSON, provenance (always). Where each
  parameter value came from, and the notices of the resolution.

#### `obs`

`--obs`, a cell's simulated observations, a by-product of the cell. Contract
version 1.

- `<stream>`: `<stream>.tsv`, TSV (each stream). One stream's simulated
  observations: `time` and the stream's columns.
- `failure`: `failure.toml`, TOML (in place of the streams when they could not
  be drawn). The error that kept the observations from being drawn.

#### `event_log`

`--event-log`, a by-product of the cell. Contract version 1.

- `event_log`: `event_log.tsv`, TSV (when it was recorded). Every event of the
  cell's run under the recorder.
- `failure`: `failure.toml`, TOML (in its place when it could not be). The error
  that kept the event log from being recorded.

#### `sim_quantities`

`--quantities-out`, a by-product of the cell. Contract version 1.

- `quantities`: `quantities.json`, JSON (when they were evaluated). The model's
  quantities evaluated on the cell.
- `failure`: `failure.toml`, TOML (in their place when they could not be). The
  error that kept the quantities from being evaluated.

#### `profile_point`

One point of a profile. Contract version 1.

- `mle`: `mle.toml`, TOML (always). `final_loglik`, a `[focal]` table with the
  profiled parameter's value, and an `[mle]` table with the other estimates.
- `provenance`: `provenance.json`, JSON, provenance (always). The point's
  settings and starts as resolved.

<!-- End of the generated result-file contract. -->

### 2.10 The output manifest

`--manifest PATH` makes a command that stores results also write a small JSON
file at PATH, the output manifest, listing the run id of every result it
produced or reused and each result file under its logical name (§2.9) with its
SHA-256 and size. A store result's folder is named by a hash, so a workflow tool
such as make or Snakemake cannot name it before the command runs; a rule
declares the manifest as its output instead, and a later step reads run ids and
file names from it. The design is
`docs/dev/proposals/2026-11-23-workflow-bindings.md`.

`simulate`, `fit run`, `pfilter`, `profile`, `predict`, `score`, `import` and
`compare` take it:

```bash
camdl fit run fits/m4w_mh.toml --label m4w_mh --manifest build/fits/m4w_mh.json
camdl import pull-staging/ --label m4w_mh --manifest build/fits/m4w_mh.json
```

`pfilter` refuses it with `--no-store`, and `simulate` with `--dry-run` or
`--design-from`, since none of them keeps a result in the store; clap exits 2.
`compare` reads and stores its predictions and scores in the stores of its
inputs, and a manifest names results in one store, so `compare --manifest` is
refused with exit 1, naming the stores, when the compared results sit in more
than one, or when no input is a stored result.

#### When it is written

After its arguments have parsed and before its first call to the store, the
command removes an earlier manifest at PATH: an empty file, or a JSON file whose
`schema` begins `camdl.output-manifest/`. A PATH that is a directory, whose
directory does not exist, or that holds any other file is refused there with
exit 1, before anything is stored, and the file is left as it is
(`PATH holds a file that is not an output manifest`). When the command succeeds,
the manifest is written to a temporary file in PATH's directory and renamed over
PATH, so a reader finds no manifest or a whole one. A process killed between the
two leaves that temporary file beside PATH, named `.NAME.<pid>.<n>.<nanos>.tmp`
for PATH's file name NAME; its leading `.` keeps it out of a glob such as
`build/*.json`, and it can be deleted. A command that fails leaves no manifest,
whether it was refused, a computation failed, a member of a batch failed or was
held by another process, or the process was killed. A command whose results are
all reused from the store writes the manifest again: the same bytes, with a new
modification time.

A manifest lists only completed results. When a by-product the command asked for
was recorded as not computed while the command otherwise succeeded, such as the
by-products of a filter run with nothing to record, the command exits 1 and
writes no manifest and no links (§2.11); the message names each result that was
not stored, by kind and run id, and the results that were, the run among them,
so the missing ones can be asked for again.

#### What it holds

This is the manifest of a filter run with `--pf-health`:

```json
{
  "schema": "camdl.output-manifest/v2",
  "camdl": {
    "version": "0.1.0+…",
    "engine_id": "c6691a8e07590752508220d277d6de58",
    "ambient_key": "09c0b02809bbb777b661008ae7e0bb85efd6642aa050462a05c09b98f45f4c04"
  },
  "label": "sir_filter",
  "results": [
    {
      "kind": "quantities",
      "run_id": "e1dc4cce0127eddbf458959369fd4430c9f6f5ec618844d5325ea27494d41f8d",
      "upstream": null,
      "contract": 1,
      "attributes": {
        "name": "sir"
      },
      "files": {
        "quantities": {
          "file": "quantities.json",
          "sha256": "71dac8712e85d5466929914da2059221b5228482e2af4265cd4091626e4eaf96",
          "bytes": 32
        }
      }
    },
    {
      "kind": "model_source",
      "run_id": "fd58d4aa96790fd4ea23302c26e522938582e579e3f51271500f1e11266833de",
      "upstream": null,
      "contract": 1,
      "attributes": {
        "name": "sir"
      },
      "files": {
        "declared": {
          "file": "declared.json",
          "sha256": "5885da0260453563d16085c564b5c522f1a4e95363bfdd177740ea03c707626e",
          "bytes": 9977
        },
        "docs": {
          "file": "docs.json",
          "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
          "bytes": 2
        },
        "source": {
          "file": "source.ir.json",
          "sha256": "234228b6198339feffc1bc17e4c1c2f2d20e5d6f501fdcaf19b4fc345ebe67f0",
          "bytes": 7576
        }
      }
    },
    {
      "kind": "compiled_model",
      "run_id": "51ee63aa0bad34bdbcb8fda33789eacf11132eb7b619c37c5be9f6eeddb17083",
      "upstream": null,
      "contract": 1,
      "attributes": {
        "name": "sir"
      },
      "source": "fd58d4aa96790fd4ea23302c26e522938582e579e3f51271500f1e11266833de",
      "files": {
        "model": {
          "file": "model.ir.json",
          "sha256": "7ddca3972d177486c9336c8b495c175d97a805e4495901374a93d6e735d5bf63",
          "bytes": 3028
        }
      }
    },
    {
      "kind": "data_table",
      "run_id": "b275e1ccd19aba3819c8f2dafc4a58df9a5b618398598d7f82a071261a7369f8",
      "upstream": null,
      "contract": 1,
      "attributes": {
        "stream": "cases"
      },
      "files": {
        "table": {
          "file": "table.tsv",
          "sha256": "39ac9f3c3d89837dddda6f50d50c898919526afc11ff917eb9c2d536a0eeb179",
          "bytes": 35
        }
      }
    },
    {
      "kind": "pfilter",
      "run_id": "9babe8c932c64568e233168cda98907e91318b9855f526505089e14bfe00b3a0",
      "upstream": null,
      "contract": 1,
      "attributes": {
        "model": "sir",
        "seed": 1
      },
      "files": {
        "loglik": {
          "file": "loglik.toml",
          "sha256": "6e1b475931dfb2d1fd20ed1a5735ceefac8627f59329983bd8d675c26df37574",
          "bytes": 63
        }
      }
    },
    {
      "kind": "pf_health",
      "run_id": "d5436e03216c6d3241f18040bebf911d4dd460a38a353dbd18e9003f41353aeb",
      "upstream": "9babe8c932c64568e233168cda98907e91318b9855f526505089e14bfe00b3a0",
      "contract": 1,
      "attributes": null,
      "files": {
        "pf_health": {
          "file": "pf_health.tsv",
          "sha256": "45f16be10d3fde9fa287f8985d118b3ec99bb2ba889a561ff5460cb88c811ded",
          "bytes": 154
        }
      }
    }
  ]
}
```

- `schema` is `camdl.output-manifest/v2`. A change to the fields below changes
  the version.
- `camdl` is the build that wrote the manifest: `version` as `camdl --version`
  prints it (elided in the example above), `engine_id`, the 32-hex build
  fingerprint, and `ambient_key`, the 64-hex hash of the build and the
  environment switches that this build's run ids hold. They describe the writing
  build only. A result computed by another build, such as one placed by
  `import`, keeps its own run id, which holds that build's ambient key, not this
  one.
- `label` is the command's `--label`, or `null`. It is recorded and never read
  back as a reference: a manifest names its results by run id.
- `results` lists each distinct result the command's invocation record lists, in
  the order it first lists it. Each entry has the result's `kind`, its `run_id`
  (64 hex digits), `upstream`, the run id of the result it is a by-product of,
  or `null` for any other result, `contract`, its kind's contract version
  (§2.9), which changes whenever a file of the kind is added, removed or renamed
  or a column changes (a test pins each version to the kind's files),
  `attributes`, the attributes the invocation record gives it, and `files`. A
  `data_table`'s attributes leave out the `--data` path as typed, which would
  make `cases.tsv` and `./cases.tsv` give two manifests; they keep its `stream`.
  A `compiled_model` entry also has `source`, the run id of the model source
  (§2.9, `model_source`) the invocation recorded for it, the model file the
  command compiled or the one the result it read was computed from; that model
  source is itself an entry of the manifest. The field is absent on every other
  kind, and on a compiled model for which no model source is recorded.
- `files` maps each result file's logical name (§2.9) to `file`, its stored
  name, `sha256` and `bytes`, the digest and size the result's `run.json`
  records. Provenance and attempt files are not listed, so a recomputation under
  `--force` that changes only `execution.json` writes the same manifest.

The manifest holds no path: no store root, no result folder, and `file` is the
name inside the result's folder. A reader opens the run store as every reading
command does (§2.1) and finds each result by its run id; `camdl path` and
`camdl cat` take the manifest itself (`--manifest PATH`, §4.5) and check each
file they name against it. A manifest read against another store fails on its
first run id. The same command against two store roots therefore writes the same
manifest. It holds no time, invocation number or outcome either, so a cache hit
writes the bytes the computation wrote.

#### Which results each command lists

Every command that compiles a model file lists the compile's `quantities`, its
`model_source` and the `compiled_model`, in that order, before anything else it
stores; a command that reads a stored fit lists the fit's `model_source` and
`compiled_model`, so a compiled model's `source` always names an entry.

- `simulate`: the compile, then each cell (`sim`), each followed by the
  by-products asked for: `obs`, `event_log`, `sim_quantities`.
- `pfilter`: the compile, each bound data table, then each filter run (one, or
  one per design point) with the by-products its flags asked for.
- `fit run`: per problem, its compile and fit problem (`fit`), then the
  `fit_run`; a `--sweep` lists every problem and run.
- `profile`: the compile and fit problem, then every `profile_point`.
- `predict`: the inputs it stores, then the `prediction`. A fixed-point
  prediction stores its compile and data tables, and a forecast each `--design`
  file as a data table. A prediction from a stored fit also lists the model
  source and compiled model it read, the source of FILE under `--model FILE`.
  The fit run a prediction reads from is a dependency, not a result of the
  command, and is not listed.
- `score`: the model source and compiled model it read, the outcome data tables,
  then the `score`.
- `compare`: for each input in order, each replicate's `prediction` and then its
  `score`, whether compare read them from the store or stored them. Each has the
  attributes of the newest invocation record that lists it computed or reused,
  and the manifest's `label` is `null`.
- `import`: every result placed or found already stored, parents first, each
  with the attributes and source of the newest of the copy's invocation records
  that lists it (§4.5). A result's entry is the one the command that computed it
  wrote, apart from `camdl` and `label`, which are the importing build's and the
  import's own.

### 2.11 The link tree

`--link-root DIR` makes a command that stores results also write, once it has
succeeded, a symbolic link to each result file of each result it lists, under a
tree named by the command's `--label`. The links are for a person browsing
results, or a program that wants plain paths to open. A workflow step that must
read exactly what an earlier step made depends on that step's output manifest
(§2.10), whose run ids and digests do not move when a label is reused; the links
follow the label. The design is
`docs/dev/proposals/2026-11-23-workflow-bindings.md`, "Links".

`simulate`, `fit run`, `pfilter`, `profile`, `predict`, `score` and `import`
take it:

```bash
camdl fit run fits/m4w_mh.toml --label m4w_mh --link-root results
camdl simulate measles.camdl --seeds 1:2 --label pair --link-root results
```

```text
$ ls results/fits/m4w_mh/
chain_0  chain_1  chain_starts.tsv  diagnostics.json  draws.tsv  fit_state.toml  …
$ ls results/sims/pair/
19d4817e5869  825d02b237bd
```

#### The root

The link root is `--link-root DIR`, else the environment variable
`CAMDL_LINK_ROOT`, else none, and with none the command writes no links. An
empty value names no root. A relative root is relative to the directory the
command runs in, as a relative store root is (§2.1). A project can set the
variable once in an `.envrc` file, which direnv reads after `direnv allow`.
Written as `$PWD/results`, the root is the project directory's `results/`
whichever of its subdirectories a command runs in, since direnv evaluates the
file in the directory that holds it:

```bash
# .envrc
export CAMDL_STORE="$PWD/camdl-store"
export CAMDL_LINK_ROOT="$PWD/results"
```

`simulate --dry-run`, `simulate --design-from` and `pfilter --no-store` keep no
result in the store: `--link-root` is refused there by argument parsing (exit
2), and `CAMDL_LINK_ROOT` is ignored. Symbolic links are written on Linux and
macOS; on any other platform a command given a link root is refused with exit 1
before it stores anything.

#### The layout

Each link is at

```text
<root>/<group>/<label>/<logical name>.<ext>
```

- The group is the result's kind's family: `fits` for a `fit_run`, `sims` for a
  `sim` cell, `filters` for a `pfilter` run, `predictions`, `profiles` for a
  `profile_point`, and `scores`. A content artifact (`compiled_model`,
  `quantities`, `data_table`) and a fit problem (`fit`) get no links.
- The label is the command's `--label`. Without one, each result's directory is
  named by its run id's first 12 hex digits.
- The file's name is its logical name (§2.9) followed by the extension of its
  stored name, so `chain_0/trace` links as `chain_0/trace.tsv` and a
  `prequential` by-product's `joint` as `joint.tsv`. A logical name holding a
  `/` makes a subdirectory.
- A by-product's files sit with its upstream's: a filter run's `pf_health.tsv`
  links as `<root>/filters/<label>/pf_health.tsv`, and a cell's observation
  stream `cases` as `<root>/sims/<label>/cases.tsv`. A by-product file whose
  name the upstream already uses sits under the by-product's kind instead
  (`obs/traj.tsv` for a stream named `traj`).
- When a command's results include several of one group, each sits one level
  deeper, under its run id's first 12 hex digits:
  `<root>/sims/pair/19d4817e5869/traj.tsv`.

A link's target is relative when the store root and the link root, with every
symbolic link in them resolved, share a directory below the filesystem's root,
so that moving that directory, such as the project, keeps the links working;
otherwise it is absolute.

Each link directory (`<root>/<group>/<label>`) holds `.camdl-links.json`, the
record of the directory's name as camdl gave it (the label, or the run id's
digits) and of the links camdl wrote in it, each with its target. The record is
how camdl tells its own links from anything else at the path. The name is
compared byte for byte: two labels that the filesystem resolves to one
directory, such as `Up` and `up` on a case-insensitive volume (the default on
macOS), or two Unicode normalizations of one word, never take each other's
links.

#### When links are written and removed

The links under a root follow the latest command with the label that succeeded
with that link root. A command run without a link root, or with another one,
leaves an earlier tree under this root in place, so its links can name the
results of an earlier invocation than `@label` does (§4.5).

- After its arguments have parsed and before its first call to the store, the
  command removes every link directory of its label, in every group, at the
  point where it removes an earlier manifest (§2.10). A command that then fails,
  in any of the ways that leave no manifest, leaves no links for the label.
- When the command succeeds, cache hits included, it writes its links from the
  results its output manifest lists, before the manifest. Every link directory's
  place is checked before any is written, and a write refused part way removes
  the directories it wrote, so a refused write leaves no links. Each link
  directory is built under a temporary name beside its place and renamed into
  it, so a reader finds no directory or a whole one, never links of two commands
  mixed. Each link is checked to resolve to its stored file. A command whose
  manifest cannot be written removes the links it wrote.
- A directory camdl replaces or removes is first renamed aside and then checked,
  so what is checked is what is removed; one that fails the check is renamed
  back.
- A link whose target the store no longer holds dangles: opening it fails.
  Running the command again, a cache hit, writes the link again.

A temporary directory left by a killed process is named `.<label>.camdl-new.…`
or `.<label>.camdl-old.…`, beside the link directory, with the process's id
after the second `.`. Its leading `.` keeps it out of a listing. A later command
with the label removes those whose process is no longer running on this host,
when they hold only what camdl wrote, and refuses one that holds anything else.

Links are an export: they are never part of an identity, a command given a link
root reuses what the same command without one stored, and the output manifest
does not mention them.

#### What is refused

A command refuses, with exit 1 and before anything is stored:

- a link root that is not a directory;
- a label that cannot name a directory: empty, beginning with `.`, holding a
  `/`, or longer than 207 bytes, which leaves room in the 255-byte limit on a
  file name for a temporary directory's suffix;
- anything camdl did not write at one of its label's link directories, or at one
  of its temporary directories: a file, a symbolic link, a directory holding
  anything other than its record and the links the record names with the targets
  it records, or a directory whose record names another label. The message names
  the path, and for another label both labels; the path is left as it is.

An empty directory at a link directory's place is not refused: it is removed.

Once the command has stored its results, its links are refused, with exit 1 and
no manifest, when the link root or a group directory, with every symbolic link
resolved, is the run store's root or lies inside it (a store inside the link
root is allowed); when a place holds anything camdl did not write, as above,
which for a command without a label is checked only here; and when a written
link does not resolve to its stored file.

---

## 3. Core Types

A run is a point in three axes — **M** (parameter values), **σ** (scenario), and
**S** (seed) — and the job of this section is to say which type owns which axis,
where a value stops being a request and becomes a resolved fact, and how that
resolved fact becomes a directory on disk.

Three type families do the work, and conflating them is the usual source of
confusion:

| Family              | Purpose                                      | Lives in                           |
| ------------------- | -------------------------------------------- | ---------------------------------- |
| **Surface types**   | Parse the CLI surface: clap args             | `rust/crates/cli/src/args/`        |
| **Engine input**    | The resolved job the surface lowers to       | `rust/crates/cli/src/sim_job.rs`   |
| **Identity values** | The fully-resolved inputs a cell is keyed by | `rust/crates/cli/src/sim_store.rs` |

The pipeline is:

```text
camdl simulate <args>  ─→  SimulateJob  ─→  engine::run_job  ─→  per-cell SimRun
                            (resolved)        (cell grid)          (resolved θ, σ, seed)
                                                                            │
                                          SimIdentity  ←────────────────────┘
                                          (model, config, params, scenario, seed)
                                                    │
                                          run id  →  the cell's folder (§4.4)
```

`SimulateJob` deliberately has no serde derives: the surface _lowers to one
resolved struct_, and a wire format for that struct would be a second schema to
keep in step with the flags.

### 3.1 SimulateJob — the resolved engine input

`rust/crates/cli/src/sim_job.rs`

```rust
#[derive(Debug, Clone)]
pub struct SimulateJob {
    /// Resolved IR/`.camdl` path, verbatim from the CLI.
    pub model: String,
    /// `--params` files, applied in order.
    pub params_files: Vec<String>,
    pub backend: ForwardBackend,
    pub dt: f64,
    /// Optional CLI `--integrator` override (method only); `None` → the
    /// model's declared integrator.
    pub integrator: Option<crate::args::types::IntegratorArg>,
    /// Where parameter vectors come from (the central dispatch).
    pub source: ParamSource,
    /// σ layer — which scenarios to run. Empty ⇒ the single implicit arm, `(none)`.
    pub scenarios: Vec<ScenarioRef>,
    /// The resolved `--to` horizon override (model time), applied to every cell
    /// after the scenario horizon; `None` = no override.
    pub t_end_override: Option<f64>,
    /// Times every cell must record a snapshot at, beyond the model's own
    /// output schedule (`camdl predict` supplies its forecast rows' closing
    /// boundaries; empty everywhere else).
    pub required_output_times: Vec<f64>,
    /// The loaded `--init-state` ensemble, shared by every cell; `None` = the
    /// model's own `init {}`.
    pub init_state: Option<std::sync::Arc<InitStateSource>>,
    /// The run's resolved observation window, folded once for the whole job so a
    /// sweep cannot resolve a different `last_obs` per cell.
    pub obs_anchors: Option<ir::anchor::ObsAnchorTimes>,
    /// S layer.
    pub seeds: Seeds,
    /// `--param NAME=VALUE` CLI overrides merged on top of every cell
    /// (M layer, highest precedence).
    pub cli_overrides: Vec<(String, f64)>,
    /// A `--draws prior|<file> --fit FIT.toml` config's `[fixed]` block,
    /// resolved at the `[fixed]` tier of every cell; empty for every other
    /// source.
    pub fit_fixed: IndexMap<String, f64>,
    /// `--param-vec PREFIX=FILE` entries (CLI only).
    pub set_vec_entries: Vec<(String, String)>,
    /// `--table NAME=FILE` entries, and under `--draws … --fit` the fit's
    /// own bound files beneath them.
    pub table_files: Vec<(String, String)>,
    /// Synthetic-observation output mode.
    pub obs: ObsOutput,
    /// Rayon thread count for the simulation phase (1 = sequential).
    pub parallel: usize,
}
```

It carries no output directory, no `--force`, no label, and no GeoJSON path:
those are write-side concerns owned by the sink, not by the job.
`camdl
simulate` builds one in `main.rs` with `parallel` from `--parallel` /
`CAMDL_PARALLEL` (absent: sequential; 0: every core); `camdl predict` builds one
for its free-forward replay and passes its own thread count.

The engine that consumes it is `rust/crates/cli/src/engine.rs`:

```rust
pub fn plan_grid(job: &SimulateJob) -> (Vec<CellSpec>, Grid);
pub fn run_job(job: &SimulateJob, sink: &mut dyn RunSink) -> Result<(), String>;
```

`plan_grid` expands the job into an ordered list of cells — the Cartesian
product `scenario × param-point × replicate`, in exactly that iteration order —
and `run_job` runs them (sequentially, or via Rayon when `parallel > 1`) and
merges each into a `RunSink` in canonical order. `simulate` supplies a sink that
commits a content-addressed leaf per cell and accumulates the combined
wide-format TSV.

Per-cell seeds are order-independent by construction, so parallelism never
perturbs a trajectory:

```text
process_seed = if explicit --seeds { seeds[rep] }
               else if total_runs == 1 { base_seed }
               else { base_seed ^ (seed_point_idx * SEED_MIX_DRAW)
                                ^ (rep            * SEED_MIX_REP) }
obs_seed     = process_seed ^ SEED_MIX_OBS
```

`seed_point_idx` is the point (draw-row) index for every parameter source except
a sweep, and is always 0 for a `--sweep` / `--sweep-file` design: a sweep point
is a value the user chose, not a sample, so every design point at a given
replicate shares one random stream (paired differences across points, and
inserting a point leaves the others' trajectories unchanged; §5.5).

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

Three accessors keep the interpretation in one place: `suppresses_trajectory()`,
`file_path()`, `dir_path()`.

CLI mapping:

```text
--obs cases.tsv          → ObsOutput::File("cases.tsv")
--obs-dir obs/           → ObsOutput::Dir("obs/")
--obs-only cases.tsv     → ObsOutput::OnlyFile("cases.tsv")
--obs-only-dir obs/      → ObsOutput::OnlyDir("obs/")
```

These four flags are mutually exclusive at the clap level. All of them are
exports: the observation draws are stored in each cell as an `obs` by-product
(§10.5), and the file or directory is rendered from them.

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
    /// Deterministic grid: the design points, each with the id it is written under.
    Sweep { points: Vec<SweepPoint>, replicates: usize },
    /// Pre-resolved parameter draws (posterior file / prior / uniform).
    Draws {
        rows: Vec<IndexMap<String, f64>>,
        replicates: usize,
        /// `Some(path)` iff the draws came from a user-authored file
        /// (`--draws <file.tsv>`); `None` for generated draws.
        explicit_file: Option<PathBuf>,
    },
}

/// One design point: the id it is written under (a sweep file's own column,
/// else its 0-based position) and the parameter values it sets.
pub struct SweepPoint {
    pub point_id: usize,
    pub overrides: IndexMap<String, f64>,
}
```

Accessors: `param_points()` returns the ordered per-cell override maps (a single
empty map for `Point`), `point_id(i)` returns the id point `i` is written under
(a sweep point's own `point_id`, else `i`), and `replicates()` returns the
replicate count. Total cells are `|scenarios| × |param_points| × replicates`,
except that an explicit seed list overrides the replicate count — see §3.5.

Which front end produces which variant:

| Variant | Produced by                                                         |
| ------- | ------------------------------------------------------------------- |
| `Point` | `simulate` with no `--draws` (`--replicates N` sets the count)      |
| `Sweep` | `simulate --sweep …` / `--sweep-file FILE` — the design axis (§5.5) |
| `Draws` | `simulate --draws …` (all four sources of §3.4)                     |

`explicit_file` is what distinguishes a user-authored draws file from generated
draws, and the distinction has teeth. A scenario whose `set` or `scale` touches
a parameter that a user-authored coordinate varies — a `Sweep` point set, or a
`--draws <file.tsv>` column — is refused before any cell runs, naming every
colliding parameter, both sources, and the fix (the diagnostic is shown in
§3.6). Over generated draws the same overlap is allowed: a scenario applied to
prior, uniform, or posterior draws is the counterfactual those runs exist for.
The check is `engine::check_scenario_coordinate_collision`, reached from
`engine::run_job` and from `simulate --dry-run` (so a dry run refuses what the
run would); `camdl predict --sweep` words its own check with the same formatter.

This is collision policy, not precedence. The resolution order of §1.3 still
places a scenario above a draw row or sweep point, and that order is what
decides distinct parameters and generated draws. It never silently discards
values the user listed.

### 3.3 Sweep specification

One sweep grammar serves every verb that takes a grid.

**`--sweep NAME=SPEC`** — `rust/crates/cli/src/args/types.rs`, used by
`camdl simulate`, `camdl profile`, `camdl fit run` and `camdl predict`.
`simulate` and `camdl predict` share one expansion
(`rust/crates/cli/src/sweep.rs`): several `--sweep` flags form the Cartesian
product **sorted by parameter name**, the first name varying slowest, and
`point_id` is a point's 0-based position in it, so the two verbs give the same
`point_id` for the same flags. `simulate` also reads a grid from a file,
`--sweep-file FILE` (§5.5).

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
concrete rows _before_ the job exists, and the rows enter as
`ParamSource::Draws`.

| `--draws <value>` | Source                                                                                    |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `<file.tsv>`      | A TSV of parameter vectors (one row per draw, columns = parameter names).                 |
| `prior`           | Sample declared priors. Requires `-n N`. `--fit <fit.toml>` is optional.                  |
| `uniform`         | Sample uniformly from each parameter's `in [lo, hi]` bounds. Requires `-n N`.             |
| `posterior`       | A completed fit's canonical post-warm-up `draws.tsv`. Requires `--fit <fit results dir>`. |

`--draws prior` resolves priors through the same three-tier precedence chain
`camdl fit run` uses — fit.toml `[estimate].prior` first, then the model IR's
`~ <dist>` declarations, then a flat fallback. With `--fit`, the sampled
parameters are the fit's `[estimate]` names, and one of them with no usable
prior in either tier is a hard error naming the remediation. With no `--fit` it
samples the parameters that declare a `~ <dist>` prior in the model's
`parameters {}` block; every other parameter is left out of the row and resolves
per cell. Either way, a run that would sample no parameter at all (no `--fit`
and no `~` prior in the model, or an empty `[estimate]`) is a hard error: every
draw would be the same parameter vector. Do **not** describe this source as
requiring a fit config.

`--draws uniform` samples `lo + (hi - lo)·U` for each parameter that declares
`in [lo, hi]` bounds. A parameter without bounds is not drawn: it resolves per
cell from `--params`, the scenario, or `--param`, and the run is refused up
front if none of them sets it. A model in which no parameter declares bounds is
a hard error: every draw would be the same parameter vector. The error says what
is sampled (each parameter's declared bounds) and where to declare them
(`name : <type> in [lo, hi]` in `parameters {}`). This is space-filling
exploration for model debugging, not a prior.

`--draws posterior --fit <fit results dir>` reads the draws the fit's sampler
(PGAS / PMMH / MH) wrote — the method leaf's `draws.tsv`, which is post-warm-up
and thinned and carries every model parameter. Resolution is by artifact, not by
method name: a leaf has a posterior iff it wrote a `draws.tsv`, so an
optimizer-only fit (IF2 / NLopt) resolves to an error rather than to a single
point dressed up as a distribution (`rust/crates/cli/src/posterior_draws.rs`).

`--draws <file.tsv> --fit <config-or-run>` additionally supplies the fit's
`[fixed]` block to every parameter the file's columns lack — a raw posterior
trace tail carries only the estimated columns, so this fills the fixed
parameters from the fit rather than leaving them unset. The block resolves at
its own tier (2 of §1.3), as on `--draws prior --fit`: `--params` and the
scenario outrank it, and a column the file carries (a row value, tier 3.5)
outranks it. The values are not copied into the rows, so `--draws-out` writes
only the file's own columns.

**The model on the command line must be the model the parameters belong to.**
Every `--fit` form above replays a fit's parameters — its posterior draws, its
`[fixed]` block, its priors — through the model named on the command line, and
those parameters mean something only under the model they were written or
estimated for. So before anything runs, `simulate` compares the two:

- a **stored fit** (a run's folder, `@label` or a run-id prefix) is compared
  with the compiled model its problem depends on,
  `<build>/compiled_model/<model stem>-<12 hex>/model.ir.json`. A `fit.toml`
  under `--draws posterior` names no run and is refused (§2.2.1);
- a **fit config** (`--draws prior|<file> --fit fit.toml`) is compared with the
  model its `[model]` names, compiled now.

The comparison is the one `camdl predict --model` uses
(`rust/crates/cli/src/fit/fitted_model.rs`, `compare_to_fitted`): the
presentation-normalized model identity, which excludes `quantities {}` and
`contrasts {}`, the model's file name, and `--output-every` (a property of this
run, applied after the check). A model that differs is refused, naming the fit,
both identities, and the fix:

```
$ camdl simulate edited.camdl --draws posterior --fit @r0 -n 2
error: --fit @r0: edited.camdl is not the model this fit ran on (fitted model da9d008d, edited.camdl 745472eb), so its parameters would be replayed through different dynamics. Quantities and contrasts are excluded from this comparison: the difference is in the dynamics, observations, parameters or simulation settings.
```

A second line, `Fix:`, names the model file the problem stored
(`<problem folder>/model.camdl`) to run with instead, or a refit of the edited
model. The stored compile, `model.ir.json`, is itself a valid model argument, so
the fitted model can always be replayed. A model that differs only in its
quantities or contrasts runs, and says on stderr that the command line's
quantities and contrasts apply. A bare directory holding a `draws.tsv` outside
any fit has no archived model and is not checked (it is said to run "as
written").

A model anchored to observed data (`simulate { to = last_obs + 2 'weeks }`, a
scenario's `to`, a forcing's `breakpoints`) is archived by `fit run` with its
anchors resolved against the fit's `[data]`, so before the comparison the
command-line model is resolved the same way, against the fit's data as it reads
now. The fit's own source therefore replays. If that data has changed since the
fit, the anchors resolve to a different horizon and the replay is refused,
saying so: the source is byte-identical to the problem's stored copy
(`model.camdl`), and the message names the horizon each side runs to. A refusal
calls the source "edited since" only when its bytes differ from that stored
copy.

**A replay runs on the simulator the parameters were estimated on.** The same
`--fit` forms also fix the simulator: the fit's forward backend, its integration
step, and, on ODE, its integrator. Without a flag, every arm of
`--draws posterior|prior|<file> --fit` (and `--init-state fit`) runs on them:

- **backend** — a stored run's own record (`backend` in its `run_inputs.json`);
  for a fit config, the config's `[method] backend`. A fit config with no
  `[method]` names no backend, and the command's default applies.
- **dt** — a stored run's recorded `fit.meta.json` `dt` (§2.2.1), else the dt
  rule over the stored model's step and `problem.toml`'s `[config] dt`; for a
  fit config, the same rule over the command-line model and the config, which
  refuses a `[config] dt` that disagrees with the model's declared step as
  `fit run` does.
- **integrator** — the model's own, which is the fit's: the model check above
  makes the command-line model the fitted one, and the integrator is part of
  that identity.

A `--backend`, `--dt` or `--integrator` that differs from the fit's still runs,
with one stderr line naming both simulators:

```
note: @r0 was fitted on chain_binomial at dt = 0.1; this replay runs on chain_binomial at dt = 0.5 (--dt as given), so the parameters drive a different simulator from the one they were estimated on
```

Each flag replaces its own setting and nothing else. `dt` is a `[config]`
setting every backend integrates at, so a `--backend` change keeps the fit's
step. The integrator is the one setting that belongs to a single backend:
`--integrator` on a replay whose backend is not ODE is refused, since it would
select nothing yet change the run's identity. A bare directory holding a
`draws.tsv` outside any fit has no record of a simulator, and runs on the
command's own (said on stderr, with the backend and step it used).

The store path names the simulator a cell ran on (`<backend>-dt<dt>-…`), so a
replay lands under the fit's backend and step. A cell stored at another backend
or step keeps its own key and stays valid there; passing that `--backend` /
`--dt` explicitly finds it again. A prediction from a stored fit, and so each
prediction `compare` makes, runs at the fit's dt.

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

The load-bearing distinction is whether the seeds were given _explicitly_.
`Seeds::explicit()` returns `Some(&[..])` only for the list form, and the engine
branches on it: an explicit list indexes directly by replicate, so seed `N`
names the same trajectory in every run that mentions it; a single base seed
derives per-cell seeds by XOR-mixing the point and replicate indices (§3.1).
With an explicit list, the list length _is_ the replicate count — the
`ParamSource`'s own `replicates` is not consulted.

Surfaces:

| Surface                   | Result                            |
| ------------------------- | --------------------------------- |
| `simulate --seed 42`      | `Seeds::Single(42)` (default `1`) |
| `simulate --seeds 1:100`  | `Seeds::Explicit([1..=100])`      |
| `simulate --seeds 1,2,42` | `Seeds::Explicit([1, 2, 42])`     |

`--seed` and `--seeds` conflict, as do `--seeds` and `--replicates`.

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
    /// A `Named` ref to a declared preset; the preset is the source of truth
    /// for enable/disable/set/scale/compose.
    Preset { name: String },
    /// An `Inline` arm built from flags; its patch (possibly empty) applies.
    Adhoc { name: String, enable: Vec<String>, disable: Vec<String>,
            params: Vec<(String, f64)> },
}

pub fn resolve_scenario_ref(scenario: &ScenarioRef, model_preset_names: &[String])
    -> Result<ResolvedScenario, String>;
```

Resolution follows the variant, the way the engine runs it — a `Named` ref
through the preset path, an `Inline` arm through the ad-hoc path:

1. `Named`, declared by the model → `Preset`.
2. `Named`, not declared → **error**, listing the declared scenarios. No name is
   special: a model may declare a scenario called `baseline`, and
   `--scenario baseline` then means that scenario; on a model that declares
   none, it is refused like any other undeclared name.
3. `Inline` → `Adhoc`, carrying its patch. The CLI builds these for the arm a
   run has without `--scenario`: `simulate`'s, labelled `(none)` or the label
   its flags derive (§5.5), and `camdl predict`'s `fitted`, each the empty patch
   unless `--enable`/`--disable` were given.
4. `Inline` whose name is a declared scenario → **error**. A scenario reference
   is either a model preset or an ad-hoc patch, never both; the model scenario
   is the source of truth.

When no `[[scenario]]` entries are defined and no `--scenario` flag is given, a
single implicit arm, labelled `(none)`, is used. This is not a "default
scenario" — it is the absence of any scenario patch, and it hashes to the real
digest of the empty delta rather than to a literal zero.

A scenario whose `set` or `scale` touches a swept parameter — directly, or
through a preset it composes — is refused before any cell runs, by
`simulate --sweep` and its `--dry-run` alike. Precedence alone (§1.3) would let
the scenario override every sweep value, so the sweep would not vary that
parameter:

```
error: parameter `beta` is controlled by both the sweep and scenario `baseline`
  sweep:     beta = [0.2, 0.3, 0.4]
  scenario:  set beta = 0.3
The scenario would override every sweep value, so this sweep would not vary `beta`.
Fix: remove `beta` from the sweep, or use a scenario that does not touch it.
```

A `scale` on a swept parameter is refused the same way ("The scenario would
rescale every sweep point of `beta`; composing a sweep with a scenario that
scales the same parameter is not supported"), and a value that arrives through
`compose` is reported with the name of the composed preset it came from. A
`[design.*]` block's points run as a sweep and are checked the same way. A
scenario on parameters the sweep does not touch composes with it by Cartesian
product as usual.

### 3.7 From a job to a path: run identity

Output paths are never constructed ad hoc. A command resolves its inputs into
typed identity values, hashes them into per-level digests, and the store derives
the path from those digests. Two properties follow: the same inputs always land
in the same directory (so a re-run is a cache hit), and different inputs never
share a directory (so a stale result cannot be served for changed inputs).

**The cell's identity.** `camdl simulate` resolves each cell into a
`SimIdentity` (`rust/crates/cli/src/sim_store.rs`), and the run store derives
the cell's run id and folder from it. Every field is a resolved value: the
parameters are name→value after every source, the seed is the cell's process
seed (§3.5), and a non-finite value is refused before anything is hashed.

| Field      | What it holds                                                                                                                                                                          |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`    | A dependency on the stored compiled model: the model as compiled and rewritten from data (anchored times, `--integrator`, bound `--table` files), before any cell's values are applied |
| `config`   | backend, `dt`, the cell's start and end, a restored `--init-state` row, `allow_degenerate_rates`, `allow_forcing_extrapolation`, `--no-flows`, the `--columns` allow-list              |
| `params`   | Every parameter's value after all sources: the model, `--params`, a draw or sweep point, the scenario, `--param` and `--param-vec`                                                     |
| `scenario` | The interventions the cell's scenario selection keeps                                                                                                                                  |
| `seed`     | The process seed                                                                                                                                                                       |

Two rules govern membership:

- **A field that changes the stored bytes is identity and must re-key.** This
  includes output-view flags: `--no-flows` and `--columns` change which columns
  the cell stores, so they are in `config`. `--output-every` is not there
  because it lowers into the model's output schedule and so changes the compiled
  model instead.
- **A re-encoding of the same values is presentation and enters no hash.** A
  rendering choice, such as a file format or `--dates`, has no field in the
  identity.

The run id hashes the kind, the ambient key (the camdl build and the environment
switches that change results) and the identity's key. The folder groups the cell
by model, settings, parameters and scenario, each a `{label}-{12 hex}` name,
under the build directory `<root>/<version>-<12 hex of the ambient key>/`:

```text
camdl-store/0.1.0_ed976bd25-5ac7c5a662df/
  sim/
    sir_basic-f7f8d6574383/               # the model
      chain_binomial-dt1-e11ea3bb8bda/    # the settings
        base-883acc24d0c8/                # the parameters
          _none_-0982c8c677d0/            # the scenario (here none)
            seed_42-4efed5f6564d/         # the cell: the seed typed
              identity.json
              run.json
              traj.tsv
```

The labels (§2.4) are provenance: a cell is found by its identity, and a folder
name is never parsed. Navigation and display read the records.

## 4. Single-Run Simulation

The `.camdl` file plus a parameter TOML is a complete, self-contained
specification for running a model. No fit config is needed for exploration.

### 4.1 Basic CLI

```bash
# Baseline (no scenario) — stores the cell in the run store
# and prints where it landed on stderr (§4.4)
camdl simulate seir_vaccine.camdl --params params.toml --seed 42
```

```
stored /home/me/project/camdl-store/0.1.0_3e2b2888-4b1f07d2a9e3/sim/seir_vaccine-6d02b7e914fa/chain_binomial-dt1-5e2a90c417bd/base-77c0d1e25f3a/_none_-0c4d21e9b7a3/seed_42-e81f5a0c3d29
       camdl cat e81f5a0c3d29c1b04f6e7a2d9b3c58e01f4a6d27c9b8e3f05a1d6c4b2e7f9a03
```

The hex suffixes and the `run_id` are content hashes; they change whenever any
identity-bearing input changes (including the camdl build), so the exact digests
in this document will not match your run.

```bash
# Named scenario from the model's `scenarios {}` block
camdl simulate seir_vaccine.camdl --params params.toml --scenario with_sia --seed 42

# Ad-hoc intervention toggle (no named scenario)
camdl simulate seir_vaccine.camdl --params params.toml --enable sia_round_1 --seed 42

# Parameter override (M layer)
camdl simulate seir_vaccine.camdl --params params.toml --param beta=0.5 --seed 42

# Scenario + parameter override (σ layer + M layer); valid because `with_sia`
# sets no parameter — a `--param` on a parameter the scenario sets is refused
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
  scenario: (none)

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
still deterministic and still stored. A single-cell `--dry-run` prints the plan
alone; a multi-cell one also lists which cells the store already holds (§5.5).
Neither writes anything.

### 4.2 CLI Flag Rules

**`--scenario` and `--enable`/`--disable` are mutually exclusive (both σ
layer).** clap rejects the combination at parse time, before the model is
compiled:

```bash cli-check=refused
camdl simulate seir_vaccine.camdl --params params.toml --scenario with_sia --enable sia_round_1
```

```
error: the argument '--scenario <SCENARIOS>' cannot be used with '--enable <ENABLE>'

Usage: camdl simulate --params <FILE> --scenario <SCENARIOS> <MODEL>

For more information, try '--help'.
```

To combine a preset with an extra toggle, define a composed scenario in the
model file.

**`--param` composes with a scenario that does not set or scale the same
parameter** (M layer beside σ layer); on a parameter a selected scenario sets or
scales it is refused (§1.3, gh#571):

```bash
camdl simulate seir_vaccine.camdl --params params.toml --param beta=0.5 --seed 42
camdl simulate seir_vaccine.camdl --params params.toml --scenario with_sia --param beta=0.5 --seed 42
```

**Other exclusions enforced at parse time:**

| combination                                                                                                                               | why                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `--seed` × `--seeds`                                                                                                                      | one base seed vs an explicit seed list  |
| `--seeds` × `--replicates`                                                                                                                | explicit seeds vs derived seeds         |
| `--stdout` × `--seeds` / `--replicates` / `--draws` / `--sweep` / `--sweep-file`                                                          | `--stdout` is single-cell only          |
| `--stdout` × `-o` / `--obs*`                                                                                                              | `--stdout` is the one export of the run |
| `--event-log` × `--seeds` / `--replicates` / `--draws` / `--sweep` / `--sweep-file`                                                       | event-log recording is single-run only  |
| `--event-log` × `--stdout`                                                                                                                | `--stdout` is the one export of the run |
| `--draws-out` without `--draws`                                                                                                           | it writes the rows `--draws` generated  |
| `--obs` × `--obs-dir` / `--obs-only`; `--obs-dir` × `--obs-only`; `--obs-only` × `--obs-only-dir`; `--obs-only` / `--obs-only-dir` × `-o` | one observation output form at a time   |

The `--obs*` combinations are clap conflicts like the rest (§3.1.1), rejected at
parse time. Two checks need the compiled model and run after it, exiting 1 with
a one-line message: `--fit` is legal without `--draws` only when something needs
the fit's data or config rather than its posterior — an anchored `--to`, an
anchored model, or a config with `[simulate] from` — and is refused otherwise
(`--fit requires --draws (or an anchored --to, an anchored model, or a config
with [simulate] from)`).
`--init-state fit` is a separate check: it needs
`--draws posterior --fit <fit results dir>`.

**Precedence of parameter sources (last wins).** Compiled-in values (§1.3 tier
1, normally none) → `--params` files in order → draw row or sweep point →
scenario `set`/`scale` (preset or inline) → `--param`. A scenario that `set`s a
parameter therefore overrides a value coming from generated `--draws`. `--param`
outranks a scenario only on parameters the scenario leaves alone; a `--param` on
a parameter a selected scenario sets or scales is refused (§1.3). A scenario
touching a parameter that a `--draws <file>` column or a `--sweep` varies is
refused rather than resolved (§3.2). See §5.4.

### 4.3 Parameter and Model Inputs

```bash
--params FILE              # load a parameter TOML (may be repeated; applied in order)
--param NAME=VALUE         # override a single parameter (may be repeated)
--param-vec PREFIX=FILE    # override an indexed parameter family from a file
--table NAME=FILE          # supply the file for an external input `external("NAME")`
```

A model declaring `C : patch × patch = external("contact")` requires the
matching `--table`:

```bash
camdl simulate sir_ext_table.camdl --scenario baseline --seed 1
```

```
error: external input 'contact' has no file: table 'C' reads it.
  Fix: bind one with `--table contact=FILE`, or in the fit config `[data.tables] contact = "FILE"`.
```

```bash
camdl simulate sir_ext_table.camdl --scenario baseline --seed 1 --table contact=contact.tsv
```

The table file is a headerless numeric TSV holding one number per cell of the
declared dimensions; a file with more or fewer is refused. The same flag binds a
forcing declared `data = external("NAME")`, whose file is read as a
`data = "path"` forcing's is (language spec §7). A `--table` naming an input the
model does not declare is refused. The file's bytes enter the run identity, so
editing it produces a new leaf. `pfilter`, `profile` and `simulate --fit` take
`--table` over the fit config's `[data.tables]`
([`docs/fit-toml.md`](fit-toml.md)).

**Backend and integration knobs.**

```bash
--backend {gillespie|chain_binomial|ode}   # default chain_binomial
--dt N                                     # step size for discrete-time backends (default: the model's dt, else 1.0)
--integrator {rk4|rk45}                    # ODE method override, refused off ode; tolerances live in the model
```

**Trajectory column view.** These change the emitted columns _and_ the run
identity, so a filtered run is a distinct cell:

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
# 0.1.0+3e2b2888
replicate	scenario	seed	time	t_start	t_stop	S	I
0	(none)	42	0	0	0	99990	10
0	(none)	42	30	0	30	99791	64
0	(none)	42	60	30	60	96434	1208
```

`--dates` is presentation: it adds the date columns to the rendered table and
does not change the stored cell.

Without an `origin` in the model, `--dates` is an error:

```
error: --dates requires the model to declare an `origin` (e.g. `origin = date("2020-01-01")`).
```

### 4.4 Output destinations — the run store by default

`camdl simulate` stores every cell in the run store, whose root is found as §2.1
states, and reports the folder plus the `camdl cat` that reads it back. Both
lines go to **stderr**, so a `--stdout` pipeline stays clean. Storing is not
opt-in.

```bash
# Default: store the cell, print where it went.
camdl simulate seir_vaccine.camdl --params params.toml --seed 42
#    stored …/camdl-store/0.1.0_3e2b2888-4b1f07d2a9e3/sim/…/seed_42-e81f5a0c3d29
#           camdl cat e81f5a0c…

# Store the cell and print the combined table to stdout.
camdl simulate seir_vaccine.camdl --params params.toml --seed 42 --stdout > out.tsv
camdl simulate seir_vaccine.camdl --params params.toml --seed 42 --stdout | head

# Store the cell and export the combined table to a file.
camdl simulate seir_vaccine.camdl --params params.toml --seed 42 -o out.tsv
#   trajectory written to out.tsv

# Browse what you have run.
camdl list
camdl show <run-id-prefix>
camdl cat  <run-id-prefix>
```

**The three destinations.** Each says where the combined table goes; every one
stores the cells.

| flag                        | combined table                    |
| --------------------------- | --------------------------------- |
| _(default)_                 | not written; the cells are stored |
| `-o FILE` / `--output FILE` | `FILE`                            |
| `--stdout`                  | stdout                            |

The trajectory is **never** written to stdout without `--stdout`, at any size.
Commands whose primary result is a scalar do echo it: `camdl pfilter` prints the
log-likelihood to stdout while also storing its result, so the tight "vary θ,
re-check" loop stays a one-liner:

```bash
camdl pfilter seir_observations.camdl --scenario baseline \
    --data weekly_cases=obs/weekly_cases.tsv --data detection=obs/detection.tsv \
    --particles 200 --seed 1 2>/dev/null
```

```
-190.8639
```

**The combined table.** `-o`, `--stdout` and `camdl cat @LABEL` all write one
table rendered from the stored cells, in the order the command asked for them.
It has a `# <version>` line, then the columns
`replicate scenario [point_id sweep:<param>…] [draw] seed time t_start t_stop [date date_start date_stop]`
and the state and flow columns. The columns are the same for one cell and for
many, so a script does not branch on the number of cells:

```
# 0.1.0+3e2b2888
replicate	scenario	seed	time	t_start	t_stop	S	E	I	R	V	flow_infection	flow_progression	flow_recovery	flow_waning
0	baseline	1	0	0	0	99990	0	10	0	0	0	0	0	0
```

`replicate` is the cell's 0-based position in the invocation, not its seed.
`scenario` is the arm's label (§2.4): the scenario's name, or for an ad-hoc arm
`(none)` or the form its flags derive, such as `(none)+vaccination`. `point_id`
and one `sweep:<param>` column per swept parameter appear under `--sweep` /
`--sweep-file` (§5.5), and `draw`, the 0-based draw index, under `--draws`.
`seed` is the process seed the cell ran under: passed back to `--seeds`, it
reproduces that cell. The date columns appear under `--dates`. A cell's stored
`traj.tsv` holds the time, state and flow columns alone, without the version
line.

**Re-running.** A cell is found by its identity (§3.7), not by its folder name,
so a renamed model file or a relabelled arm is still found. A rerun reuses every
stored cell, computes only the cells the store does not hold, and renders the
exports from the stored cells; a multi-cell run reports
`N of M cells read from the store; simulated K`. `--force` recomputes every
cell, and moves each stored one to quarantine first. `--parallel N` (env
`CAMDL_PARALLEL`) runs the cells on N threads (0 = every core; absent =
sequential); the stored cells and every export are the same bytes whatever the
value, because a cell's seed is a function of its position in the grid.

**`--label L`** names the invocation, which `camdl cat @L` and the other
commands that take a handle find again; it never names a cell. A rerun under
another label reuses every stored cell, and `@L` then names the new invocation.

**`--dry-run`** prints the resolved plan. For a multi-cell run it also lists
each cell as `[hit ]` (the store holds it, with its 12-hex id) or `[miss]`,
marks `[fail]` a cell whose identity does not resolve, and prints the counts. It
writes nothing.

**By-products.** Observations (`--obs`, `--obs-dir`, `--obs-only`,
`--obs-only-dir`, §4.6), the event log (`--event-log`) and generated quantities
(`--quantities-out`) are by-products of a cell, stored inside its folder, each
with its own `identity.json` and `run.json`:

```
seed_42-e81f5a0c3d29/
  identity.json  run.json  traj.tsv
  obs/<stream names joined by _>-<12 hex>/<stream>.tsv …
  event_log/event_log-<12 hex>/event_log.tsv
  sim_quantities/quantities-<12 hex>/quantities.json
```

Asking for a by-product never changes the cell's id or its record. A rerun that
asks for a by-product the store lacks simulates the cell again, checks that the
trajectory reproduces the stored one, and stores the by-product. The quantities
block a cell evaluates is stored once as a content artifact under
`<build>/quantities/<stem>-<12 hex>/`, as the compiled model is under
`<build>/compiled_model/`. The observation and quantity exports are rendered
from the stored by-products; the bare `--event-log` exports nothing, and
`--event-log PATH` exports the stored log.

**Failures.** A cell whose engine stops with an error is a completed result: its
folder holds `failure.toml` in place of `traj.tsv`, and each by-product asked of
it holds the same `failure.toml`. An observation draw that is refused, or a
quantity that is not finite, stores `failure.toml` in that by-product, and the
cell keeps its trajectory.

```toml
# failure.toml
stage  = "engine"          # "observations" or "quantities" in a by-product
reason = "negative propensity -3.2e-4 for transition 'recovery' at t=41.5"
```

The batch carries on past a failed cell. At the end the command lists every
stored failure, exports nothing and exits 1; a rerun reuses the stored failures
and computes nothing for them. A failure a rerun could change, such as an I/O or
store error or an interruption, leaves no folder, so a rerun computes that cell.

**Writes that store nothing.** `--design-from` (§4.6), `--projected-out` and
`--draws-out` write files from the resolved inputs and store nothing.

**Hash composition.** §3.7 lists what a cell's identity holds. `--no-flows` and
`--columns` change the settings group only. `--output-every` rewrites the
compiled model's output schedule, so it changes the model group as well: same
model source, different compiled model. `--emit-every` changes neither: it
touches only emitted observations, so it changes the identity of the `obs`
by-product (§10.5). The camdl build enters the ambient key, so a camdl built
from changed source stores new cells rather than reusing old ones.

**Several cells.** `--seeds`, `--replicates`, several `--scenario`, `--draws`
and `--sweep` store one cell per grid point; no ensemble is stored:

```bash
camdl simulate seir_vaccine.camdl --params params.toml --scenario baseline,with_sia --seeds 1:3
```

```
2 scenarios × 3 replicates = 6 runs
   stored 6 cells · /home/me/project/camdl-store
          camdl list
```

### 4.5 `camdl list` / `show` / `cat` / `path` — the primary access path

Because output goes to the store by default, these are **the** way you reach
results. They read one root, the run store (`--store DIR`, else `CAMDL_STORE`,
else `./camdl-store`); `show` and `cat` given a result's folder read the store
that folder lies in. Every run's banner hands you the exact `camdl cat` to run
next. With nothing stored, `list` prints `(no stored results)`; `show` and `cat`
refuse a handle that names nothing with
`error: no run matches 'X' in the run store <root>`.

```bash
# Tabular overview, most recent first, one section per artifact kind
camdl list
camdl list --since 1h              # last hour
camdl list --model seir_vaccine    # substring match on the model path
camdl list --scenario baseline --all
camdl list --kind sim              # sim | fit | profile | pfilter | all
camdl list --format json           # machine-readable; see the caveat below

# Full metadata for one cell (short-hash prefix resolves git-style)
camdl show e81f5a0c

# Emit the trajectory, or a named file of the cell or of its by-products
camdl cat e81f5a0c
camdl cat e81f5a0c cases_urban
camdl cat e81f5a0c event_log.tsv
camdl cat e81f5a0c sim_quantities:quantities

# The combined table of the invocation labelled `baseline`
camdl cat @baseline

# The path of a stored file, and one row per cell of a manifest
camdl path e81f5a0c traj
camdl path --manifest build/baseline.json traj --each
```

`camdl list` shows the stored cells in a `sims` table, one row per cell:

```
sims
 CREATED   RUN_ID    LABEL         MODEL      CONFIG              SCENARIO  SEED       PATH
 just now  5f62c5cb  sw            sir_basic  chain_binomial-dt1  (none)    seed_1-r1  camdl-store/0.1.0_3e2b2888-4b1f07d2a9e3/sim/sir_basic-f7f8d6574383/…/seed_1-r1-5f62c5cb63a6
 just now  4efed5f6  <unlabelled>  sir_basic  chain_binomial-dt1  (none)    seed_42    camdl-store/0.1.0_3e2b2888-4b1f07d2a9e3/sim/sir_basic-f7f8d6574383/…/seed_42-4efed5f6564d
```

`LABEL` is the label of an invocation that holds the cell. `SEED` is the cell's
seed label (§2.4). `camdl show` on a cell prints its path, kind, model, config,
scenario, seed label and process seed, full `run_id`, parameters, files and
creation time. `camdl cat` on a cell prints its `traj.tsv`, and
`camdl cat <cell> NAME` a file of the cell or of one of its by-products, such as
an observation stream (`cases`), `event_log` or `reactive_log`; on a filter run
it prints `loglik.toml`, and with a name a file of the run or of a by-product
its `--trace`, `--pf-health` and `--save-*` flags stored. Names are looked up as
the next section states. `camdl cat @L` renders the combined table of the
invocation labelled `L` (§4.4). `--limit` defaults to 50; `--all` removes the
cap.

Fits are listed one row per fit run, in a `fits` table:

```
fits
 CREATED  RUN_ID    FIT_ID    LABEL  MODEL   METHOD  SEED  BEST_LL  PATH
 4m ago   44fbdf1c  c39a2f44  r0     sir_r0  pmmh    4     -60.70   camdl-store/0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir_r0-c39a2f446d42/runs/pmmh-fc5c33ebd296/seed_4-44fbdf1c80d9
 5m ago   690f27db  c39a2f44  r0     sir_r0  pmmh    4     -60.70   camdl-store/0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir_r0-c39a2f446d42/runs/pmmh-fc5c33ebd296/seed_4-690f27db3418
```

`RUN_ID` is the run's id and `FIT_ID` its problem's. `LABEL` is the label of an
invocation that holds the run (§6.8), so both runs above show `r0`: the second
was made by an earlier invocation labelled `r0`, and is `@r0~1`. `BEST_LL` is
the run's best log-likelihood, from its `run_inputs.json`. Interrupted and
failed runs are not listed. `--history` and `--runs` change nothing in the fits
table.

Paths are printed relative to the root you asked for, so they are copy-paste
ready into `camdl show` / `camdl cat`. Short-hash prefix resolution is
git-style; in the run store a prefix that matches several results, cells and
by-products alike, is refused with their 12-hex ids:

```bash
camdl show 3
```

```
error: `3` matches 3 results: 35f6c644d453, 3766108b8da9, 3e4f1379e077
```

A fit run shown by `camdl show` prints its path, kind, full `run_id`, problem
id, method, seed, length and files; `camdl cat <run> NAME` prints one of its
files (`camdl cat @r0 draws`).

`--format json` emits the sim, fit, profile and pfilter sections as JSON Lines
(one object per line). The stream is therefore not a single JSON document; parse
it per line or filter with `--kind`. A sim or pfilter row, and the
`camdl show --format json` of one, carries `attributes`, the attributes the
answering invocation record gives the result, as the output manifest does
(§2.10). A fit row is one run:

```json
{"best_loglik":-60.69877640031929,"kind":"fit_run","label":"r0","method":"pmmh","model":"sir_r0","path":"camdl-store/0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir_r0-c39a2f446d42/runs/pmmh-fc5c33ebd296/seed_4-44fbdf1c80d9","problem_id":"c39a2f446d42c89dc3cfa679f49cdf99bd1ff0e58b296269e084a9f9d6981771","run_id":"44fbdf1c80d99fc484a400970e0183922961fc530972199790790b5425cbb2d5","seed":4}
```

A stored result's record is sealed, so a result is named when it runs, with
`--label` (§6.8). Running the same command again with `--label L` reuses its
stored results and names them `@L`.

#### Looking up a stored file: `path` and `cat`

`camdl path` prints where a stored file is, for a program that reads it, and
`camdl cat` prints its bytes. Both find the file by the rules below, from a
handle or from an output manifest (§2.10) and the file's name, and differ only
in what they print. A workflow step reads a path at once, from the manifest its
rule declares as an input, and does not record it:

```bash
camdl path --manifest build/fit.json draws            # one absolute path
camdl path @scout chain_0/trace chain_1/trace         # one path per name, in order
camdl path --manifest build/fit.json --kind fit_run --all
camdl path --manifest build/sweep.json traj --each    # one row per result
camdl cat  --manifest build/fit.json draws
camdl cat  --manifest build/sweep.json traj --each
```

A lookup is among the candidate results its source names:

- an output manifest (`--manifest M`): each entry, in the manifest's order;
- a label, `@L` or `@L~n`: each completed result its invocation's record lists,
  in that order;
- a run id, a prefix of one, or a result's folder: that result, then each
  by-product stored for it (the results whose `upstream` it is, §2.10), so a
  cell's handle reads its observation streams. Every rule below is tried on the
  result itself before its by-products, so `run.json` names the cell's record,
  and `obs:run.json` its observations'.

A label is one to 64 characters once surrounding whitespace is trimmed, each an
ASCII letter or digit, a space, a comma, a dot, an underscore or a hyphen, and
is neither `.` nor `..`. Every command that takes `--label` applies this rule
before it stores anything and records the trimmed label; it refuses any other
label, such as `f~1`, which `@f~1` would read as the label `f` and its first
previous holder:

```
error: the label `f~1` is refused: `~` at position 2 is not allowed: a label holds letters, digits, spaces, commas, dots, underscores and hyphens
```

For a label, a run id, a prefix or a folder and a name qualified by
`model_source` (`model_source:render`), the model source of the model the result
ran is a candidate too: the one the invocation that answers for the source
recorded as the `source` of the result's compiled model (§2.9, `model_source`).
For a label that is its invocation; for a run id, a prefix or a folder, the
newest invocation that listed the result computed or reused. No other invocation
stands in. When that invocation recorded no source, the name finds none and a
warning says that no model file was recorded. When an earlier invocation that
listed the result recorded a different source, an info notice names it and the
labels that name its invocations; when the answering invocation itself recorded
several, as an import of results another store's invocations fitted from several
files does, the notice names the others. A manifest lists the model source of
each compiled model it lists whose command recorded one, whether the command
compiled the model file or read a stored result (§2.10).

A run id, a prefix or a folder naming a result that has not completed is refused
for the result's own files, which are incomplete until it ends, but not for a
name qualified by `model_source`: the model source is a completed result of its
own, and the result's record names its compiled model from the start. When no
closed invocation listed the result computed or reused, the newest running
invocation that lists it answers, and an info notice says that invocation has
not ended. A failed listing answers for nothing, so a result that failed, or was
interrupted and its record closed by `camdl clean`, is refused for its model
source as it is for its own files.

`--kind KIND` keeps the candidates of one kind, and `--run PREFIX` the one whose
run id begins with `PREFIX`. A name is resolved by the first of these rules that
any candidate answers:

1. `KIND:NAME` names `NAME` among that kind's candidates: `obs:cases`,
   `fit_run:draws`.
2. A logical name (§2.9): `draws`, `chain_0/trace`, `cases`.
3. A stored name: `draws.tsv`. `identity.json` and `run.json` name the result's
   records; provenance files are named by both their names.
4. With a handle, a path through a result stored inside a candidate's folder,
   such as a prediction inside its fit run:
   `predict/forecast-<12 hex>/forecast/cases`. Each folder of the path is a
   plain name, never `..`, `.`, empty or absolute, and the nested result's
   identity must name the candidate as its parent or an ancestor. A manifest
   names only its own entries, so this rule is not tried through one.

A `KIND:NAME` whose kind no candidate is, is refused with the kinds there are. A
listed result the store holds but cannot read, damaged or unfinished, is passed
over unless it could answer the name: a lookup whose holders include a result of
its kind is refused naming it, since the answer could leave it out, and so is a
name nothing else answers.

A name more than one candidate answers is refused with the candidates, unless
`--each` is given:

```
error: 3 results of build/grid.json have `traj`: sim d71f50cbcf6c, sim 0bf1c2f2fb5a, sim 7e8f0d6c1a2b; pass --kind or --run for one, or --each for all
```

A name no candidate answers is refused with each candidate's files, by logical
name, and for a run id or a folder with each by-product of the result that is
stored but damaged or unfinished, which is not read, and the flag that stores it
again with `--force`:

```
error: the pgas run 4197734faa1d has no `mle_params`; its files are chain_0/resume_state, chain_0/trace, …
```

Through a manifest, a file's digest and size in the store's `run.json` must be
the ones the manifest records before its path is printed or its bytes are read,
or the lookup is refused, naming the file and both digests:

```
error: build/fit.json does not describe the stored file `draws` of the pgas run 4197734faa1d: the store records sha256 … and 9312 bytes, and the manifest sha256 … and 9312 bytes; run the command that wrote the manifest again
```

`cat` also checks the bytes it reads against the digest `run.json` records, and
refuses a file that differs as damaged. Given a label, both commands name on
stderr the kind and run id the label resolved to, a fit run as `run`:
`@scout → run 4197a3 (<folder>)`, `@pred → prediction 263942 (<folder>)`.
`cat HANDLE --stream NAME` is refused, saying to write `cat HANDLE NAME`.

`path --all` prints every result file of one result, as its logical name, a tab
and its path, in name order. Through a label, the one result is the one that is
not a content artifact, a fit problem or a by-product, when the invocation has
exactly one such: the fit run of a `fit run`. Otherwise `--kind` or `--run`
selects it.

`--each` takes a manifest or a label and one name, and acts on every candidate
that holds the name, in the source's order. `path --each` prints a TSV whose
columns are `run_id` (all 64 hex digits, a handle camdl reads back), one column
per attribute key (the attributes the answering invocation record gives the
result, §2.10), then `path`. The attribute columns are the union of the
candidates' attribute keys, in the order first seen, with an empty cell where a
result lacks one. A cell's attributes give `model`, `params`, `scenario`,
`replicate`, `seed`, `seed_replicate` and `process_seed`, then for a sweep point
`point_id` and one column per swept parameter, in the sweep's order, and for a
draw `draw`. A filter run's give `model` and `seed`, then for a design point
`row`, `point_id` and one column per design column. Any other kind's attributes
give one column per field, a value that is not a number, a string or a boolean
written as compact JSON. A swept parameter named like another attribute column
is written `param.<name>`, and an attribute named `run_id` or `path` is written
`attribute.<name>`. Over a three-point sweep:

```text
run_id	model	params	scenario	replicate	seed	seed_replicate	process_seed	point_id	beta	path
f24a8397841f…	sir	beta=0.2	(none)	0	1	0	1	0	0.2	/…/seed_1-f24a8397841f/traj.tsv
800fb0554c4a…	sir	beta=0.3	(none)	1	1	0	1	1	0.3	/…/seed_1-800fb0554c4a/traj.tsv
a0d44ecd5f61…	sir	beta=0.4	(none)	2	1	0	1	2	0.4	/…/seed_1-a0d44ecd5f61/traj.tsv
```

`cat --each` prints the rows of every such file as one TSV: the same attribute
columns, then the file's own columns, under one header, so each row keeps its
result's coordinates. Leading `#` comment lines are dropped, and an attribute
column named like a column of the file is written `attribute.<name>`. A file
that is not a TSV file is refused, naming its format, and so is a file whose
header differs from the first file's, naming it and both headers, before any row
is printed. Each file is read and checked as `cat` reads one.

`cat` given no name prints the result's summary file: a cell's `traj.tsv`, a
filter run's `loglik.toml`, a fit run's `fit_state.toml`, a fit problem's
`problem.toml`, a prediction's `report.json`, a score's `summary.tsv`, a profile
point's `mle.toml`, and the one result file of any other kind. `--chains IDS`
and `--exclude-chains IDS` keep the rows of the named chains of a fit run's
`draws`, and are refused for any other file.

Both commands exit 0 when every name resolved and 1 on a refusal: a handle that
names nothing (`no run matches 'X' in the run store <root>`), a manifest's
result the store does not hold, a name no candidate or several candidates hold,
a manifest that differs from the store, `--each` given a run id or a folder,
which name one result, and a `--kind` camdl does not store. `--each` with
several names and `--all` with a name are usage errors, exit 2. An attempt file
such as `progress.json` has no digest and is not looked up.

#### Maintaining the run store: `status`, `verify`, `clean`, `reindex`

Four commands read and tidy the run store, the store of `simulate` and
`pfilter`. Each takes `--store DIR`, else `CAMDL_STORE`, else `./camdl-store`,
and refuses a root that has no `store.json`, exiting 1, rather than create a
store there.

```bash
camdl status                 # runs that are running, interrupted or failed
camdl verify                 # every completed result against its records
camdl verify 318a6e1b        # one result, by run-id prefix or folder
camdl verify @baseline       # every result of the invocation labelled baseline
camdl clean                  # delete the quarantine, earlier attempts, dead staging
camdl reindex                # rebuild the index from the records
```

- **`status`** prints one line per invocation still recorded as running, and one
  per result folder recorded as running or failed, with its kind and 12-hex id.
  A running invocation or result is `running` while its owner, the process that
  claimed it, is alive, and `interrupted` once that process has stopped; a
  failed result is printed with the reason it recorded. Completed results are
  not printed. An interrupted result's folder stays until a run that asks for
  the same result takes it over. An interrupted invocation is listed, with
  `(clean closes its record)`, until `clean` closes it.
- **`verify [HANDLE]`** checks every completed result, or the results the handle
  names: an id prefix or a folder names one, and `@L` names every result of the
  invocation labelled `L`. For each it computes the run id again from
  `identity.json` and compares it with the folder's, and computes the SHA-256 of
  every file the manifest in `run.json` lists and compares it with the recorded
  digest. Each result that fails is printed as `damaged:` with its folder and
  what differs, and the command exits 1; otherwise it prints
  `every result verified` and exits 0. It also checks that the model source each
  invocation record names for a compiled model it checks (§2.9) is stored here,
  and prints each one that is not as `unresolved:`, with the invocation, the
  source and the compiled model; a folder of a kind this binary does not store,
  such as a `model_presentation` folder an older build wrote, is not checked,
  and the folders of such kinds are counted on one `not checked:` line. Neither
  damages a result, so neither changes the exit status. It writes nothing. The
  next run that asks for a damaged result moves it to `.quarantine/` and
  computes it again. That run compares each listed file's size with the record,
  and compares the SHA-256 of a file whose modification time differs from the
  recorded one with the recorded digest, so a store copied without its files'
  modification times is read as whole. A handle that names no result is refused.
- **`clean`** deletes each staging folder (`.staging/`) of an unfinished result
  whose owner was a process on this host that has stopped (a claim, or the
  folder a claim was being built in; a process id that now names a process with
  another start time counts as stopped), everything in `.quarantine/`, every
  `.attempts/` folder, and each group and build directory that is empty, and
  prints how many of each it removed. Staging whose owner is a process on
  another host is left, and printed as `left <folder>`. It closes the record of
  each invocation still running whose process on this host has stopped: the
  record says `interrupted`, a result the command had not finished is recorded
  as failed, and the report counts the invocations closed. It deletes no result
  in doing so, and never touches a completed result, a running or interrupted
  run, a run recorded as failed, or the index. A completed result leaves the
  store only after `--force` has moved it to `.quarantine/`, so the copy
  `--force` replaced can be recovered from there until `clean` runs.
- **`reindex`** reads every `identity.json` and `run.json`, writes the index of
  completed results under `.index/` as one log, removes the index logs of owners
  known to have stopped, and prints how many result folders it read. The index
  is a cache: a result is found without it.

#### Importing results from another store: `import`

`camdl import COPY` places results computed in another run store in this one.
COPY is a partial copy of the other store: its `store.json`, the result folders
wanted, each at the path it had relative to that store's root, with the parent
folders a nested result sits in (a fit run inside its fit problem, a prediction
inside its fit run), and the invocation records that listed them. `--store DIR`
names the store to place them in, else `CAMDL_STORE`, else `./camdl-store`,
which is created when absent. `--label L` names the invocation that records
them, so that `@L` names them.

```bash
camdl import pull-staging/ --label m4w_mh
```

The copy is moved into the store's `.staging/`, and every result folder in it is
checked: its record is completed and of a record format this binary reads; its
run id computes again from its `identity.json`; every result and provenance file
has the SHA-256 `run.json` records for it; it holds no file its record does not
name; an invocation record in the copy lists it; a nested result's parent is in
the copy or already in this store; every result it depends on, reads or records
in its lineage, such as a fit's compiled model and data tables, or a model
source's quantities, is in the copy or already completed in this store; and the
model source each of the copy's records names for a compiled model (§2.9) is in
the copy or already completed in this store. A folder of a kind this binary does
not store, such as a `model_presentation` folder an older build wrote, is not
placed, and the command prints `import: not placed:` with its folder and kind;
the rest of the copy is imported. A symbolic link anywhere in the copy, and a
dot-named directory inside a result other than the store's own `.staging/` and
`.attempts/`, are refused, since no check reads what they hold. A folder that
fails is refused by its path and the reason, nothing from the copy is placed,
the copy is moved back where it was given, and the command exits 1. A failure
while the folders are being placed, such as a directory of the store that cannot
be written, moves every folder already placed back into the copy, so the store
and the copy are as they were and no invocation is recorded.

What belongs to the other store's processes, not to a result, is left behind:
each placed folder loses its `.staging/`, its `.attempts/`, its lock files and
any folder holding only a lock, and the temporaries of its records.

Each result goes to the path it had in the other store, which begins with that
build's directory, so it sits beside this build's results and never among them.
A nested result whose parent is already here goes to the same path inside the
parent's folder. A run id already stored here with the same result files is left
in place and printed as `present`; one stored with other result files, or a path
held by another result, is refused, naming both folders. Provenance files are
not compared. A folder placed by another process at a result's path while the
import runs is never replaced: when it holds the same result files the result is
`present`, and otherwise it is refused, naming that folder.

One invocation records every result imported, parents first, as reused. It lists
a result once for each distinct combination of attributes and model source the
copy's invocation records gave it, so a fit run that invocations of the other
store fitted from two model files is listed with both. The other store's
`--label`s are not carried: `@L` names the import, and a label of the other
store names nothing here. `camdl list` and `@L` then see the imported results as
they see local ones, and a command run here can read one, such as a prediction
from an imported fit run. A fit run listed with model files whose quantities
differ is refused by `predict`, by its run id and by `@L` alike, since no label
here names one of the files; `predict --model FILE` names it. The command prints
one line per result, `placed` or `present`, with its kind, its 12-hex run id and
its folder. An imported copy is gone: its results are in the store and the rest
is deleted.

An import whose process is killed while it places folders leaves each folder it
placed whole, and the rest of the copy in `.staging/`, which `clean` deletes
once the process has stopped. The copy must then be pulled again; importing it
again finds the results already placed `present`.

`--manifest PATH` writes the output manifest of the imported results, placed and
present alike (§2.10), so a workflow rule can declare a pull's manifest as its
output, and `--link-root DIR` links them under the import's label (§2.11).

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

# Export the observations only, not the combined table (SBC workflows)
camdl simulate seir_observations.camdl --scenario baseline --seed 42 --obs-only-dir obs_only/
```

`--obs` writes one wide table with a column per stream, after the coordinate
columns of the combined table (§4.4): `replicate`, `scenario`, the point columns
under `--sweep` or `--draws`, and `seed`. The columns are the same for one cell
and for many:

```
replicate	scenario	seed	time	cases_urban	cases_rural
0	baseline	42	7	2	3
0	baseline	42	14	1	0
```

Each export is rendered from the observations stored in the cells (below), so
asking for `--obs-dir` after `--obs FILE` computes nothing.

`--obs` refuses a model whose streams emit on different schedules, before
simulating:

```
error: observation streams have different schedules (weekly_cases: …step: 7.0…, detection: …step: 14.0…).
A single wide TSV cannot hold multi-cadence streams.
Use --obs-dir (one file per stream, keeps trajectory) or
--obs-only-dir (one file per stream, suppresses trajectory).
```

`--obs-dir` / `--obs-only-dir` write one file per stream, after the same
coordinate columns, under the columns the stream **declared**, so the file
re-loads under its own model (gh#833): the value under the scored column's name,
the time under the `: time` column's name — or both boundaries under the
`window_start`/`window_stop` names for a stream that declares its windows per
row, each with a `date_*` twin under `--dates`:

```
win_start	win_stop	cases
0	7	2
7	14	3
```

Each row is drawn over the period the stream's `covers` assigns to its label,
and a row whose period falls outside the run is not written. Under `closing_at`
with a schedule starting at `t_start`, the row labelled `t_start` is not
written: its period `[t_start − Δ, t_start)` was never simulated. A windowed
stream cannot go into the single wide `--obs` file, which has one `time` column;
the error names `--obs-dir`.

A stratified family (`cases[p in patch]`) is one file, not one per leaf. Its
leaves share one observation `source`, which the loader binds to a single
long-format file and routes row by row to the leaf named in the `: dim` column —
so the family is written that way, named for the source, with the dim column
filled from each leaf's own stratum (gh#884):

```
week_ending	province	province_cases
7	kivu	61
7	ituri	17
14	kivu	111
```

#### `--design-from`: the rows a fit actually scores

`--design-from <fit.toml>` replaces the emit schedule as the source of a
stream's rows. That config's `[data.observations]` is loaded exactly as
`fit run` loads it, and each stream is written on the rows the loader bound: its
observed labels, each row's own period — per-row `window_start`/ `window_stop`
widths included — and each `NA` hole, written as a row with its period and no
value. Only the values are simulated.

This is what a simulation-based self-consistency test needs (gh#831). Simulating
on a regular grid gives the synthetic fit more information than the real one
has: a three-day row becomes three one-day rows, and a hole becomes an
observation.

The temporal columns are written in the representation the bound file used. A
stream whose cells were ISO dates is re-emitted as ISO dates, rendered through
the model's `origin` and `time_unit`; one whose cells were numbers keeps the
numbers. So the synthetic replicate is the same shape as the file it stands in
for, column names and cells alike, and pairs with it directly.

```
camdl simulate model.camdl --params theta.toml \
    --design-from fit.toml --obs-only-dir synth/
```

It requires `--obs-dir`, `--obs-only-dir` or `--projected-out` (one file per
stream), and takes θ from `--params`/`--param` — one dataset, written into the
directory — or from `--draws`, which writes one `ds_NN/` subdirectory per draw
so each dataset stays a file set the loader can bind on its own. What it
produces is a **dataset, not a run**: no trajectory and no store leaf are
written, with `--obs-dir` as with `--obs-only-dir`. It conflicts with the flags
the store-backed trajectory pipeline owns (`--obs`, `--obs-only`, `-o`,
`--stdout`, `--seeds`, `--replicates`, `--emit-every`, `--dates`,
`--init-state`, `--quantities-out`, `--event-log`), and with `--scenario`, since
the dataset is written from the parameter vector alone with no scenario applied;
so nothing passed is silently dropped. To simulate a dataset under a scenario's
parameter values, give the values through `--params`.

A stream whose likelihood reads a data column — a binomial denominator
`n = tested`, a person-time offset — is refused by name: there is no data file
to read that column from when the data is what is being generated, and writing
`0` would assert an observation the run never made (gh#829). A stratified
(long-form) stream is refused for the same reason in a different shape: its
family shares one file per `source`, which one file per stream would not
re-load.

**Projections on the design: `--projected-out DIR`.** With `--design-from`,
`--projected-out DIR` writes each stream's _projected_ value on the same rows:
the model quantity the stream's `projected = …` expression defines over each
row's period, before any observation noise. One file per stream holds the
stream's temporal column(s) and a column named `projected`, at full round-trip
precision. A hole in the data still gets its value, because the projection
exists whether or not the row was observed. It may be passed with or without
`--obs-dir` / `--obs-only-dir`, and with `--draws` it writes one `ds_NN/`
subdirectory per draw.

```
camdl simulate model.camdl --draws perturbations.tsv \
    --design-from fit.toml --projected-out projections/
```

A projection reads no likelihood argument, so the data-column refusal above
(gh#829) does not apply: a stream whose dispersion or denominator comes from a
data file is projected like any other. Nor do the round-trip refusals, because
the file is not a dataset: its column is `projected`, not the stream's observed
name, so it cannot be bound back as data by accident. The value is the
projection itself; a likelihood may transform it (`mean = 0.86 * projected`).
Projections are what a sensitivity or identifiability analysis differentiates,
and what a posterior-predictive mean averages. Like `--quantities-out`, the
output is a sidecar outside run identity.

`--obs-only` / `--obs-only-dir` leave out only the combined-table export; the
cell still stores `traj.tsv`. In every mode the sampled observations are an
`obs` by-product of each cell, stored inside it at
`obs/<stream names joined by _>-<12 hex>/<stream>.tsv` with its own
`identity.json` and `run.json`, so they are reachable with
`camdl cat <id> <stream>` without keeping the exported files. A cell already
stored without them is simulated again to draw them, and its trajectory is
checked against the stored one.

These writers — `--obs`, `--obs-dir`, the stored `obs` by-product, and a
generated quantity that reduces `observations.<stream>` — draw every stream from
the model alone, with no data columns. So a stream whose likelihood reads a data
column is refused by name before any cell runs, including a ratio stream's `n`
that `--design-from` writes from the model: a missing column would otherwise be
drawn as `0` (gh#829). A model with such a stream emits none of its streams this
way; `--design-from` draws only the streams the fit binds.

The observation RNG is seeded independently of the process RNG, so adding
`--obs` does not change the trajectory.

---

## 5. Ensembles and Sweeps

One engine behind every grid. `camdl simulate` covers a grid expressed entirely
in flags: scenarios × parameter points × replicates or seeds, where the
parameter points are a sample (`--draws`) or a design (`--sweep` /
`--sweep-file`, §5.5). It builds a `SimulateJob` and stores one `sim` result per
cell in the run store (§4.4), so `camdl list` / `show` / `cat` browse every cell
uniformly. No combined table is stored; `-o` and `camdl cat @LABEL` render it
from the cells. A grid the store already holds is read back, not re-simulated.

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
camdl simulate sir_priors.camdl --draws posterior --fit @posterior --replicates 10

# ── Record the sampled parameter vectors ─────────────────
camdl simulate sir_priors.camdl --draws uniform -n 5 --draws-out draws_uniform.tsv

# ── Deterministic grid over a parameter (the design axis) ─
camdl simulate sirv.camdl --params params.toml \
    --scenario baseline,with_vaccination --sweep "beta=lin(0.2,0.6,5)" --seeds 1:50

# ── Two swept parameters: the name-sorted Cartesian product ─
camdl simulate sir_basic.camdl --params params.toml \
    --sweep beta=0.2,0.3 --sweep gamma=0.1,0.2 --replicates 10

# ── Design points from a file (one row per point) ─────────
camdl simulate sir_basic.camdl --params params.toml --sweep-file design.tsv --replicates 5

# ── Plan the grid against the store without running ───────
camdl simulate sirv.camdl --params params.toml --sweep "beta=lin(0.2,0.6,5)" --seeds 1:50 --dry-run
```

`-n` / `--n-draws` sets the draw count for `--draws prior|uniform` (there is no
`--n`). `--fit` is required for `--draws posterior`, optional for
`--draws prior` (a `fit.toml` supplying priors) and for `--draws <file>` (a
`[fixed]` block supplying the parameters the file's columns omit).

`--draws prior` samples only the parameters that declare a prior. A parameter
with no prior that nothing else sets — no compiled-in value, `--params`,
scenario, or `--param` — is refused before any cell runs, naming the scenario
that leaves it unset:

```
error: parameter 'c' has no prior, and nothing gives it a value for scenario 'other'.
  --draws prior samples only the parameters that have a prior; every other parameter takes its value from --params FILE, the scenario's `set`, or --param NAME=VALUE.
  Fix options: declare a prior on 'c' in the model's parameters {} block (`c : <type> in [lo, hi] ~ <dist>(...)`), supply `--fit FIT.toml`, set it with --params FILE or --param c=VALUE, or set it in scenario 'other'.
```

A prior whose mass falls partly outside the declared bounds is rejection-sampled
and reported:

```
warning: prior for 'gamma' placed 16.7% mass outside declared bounds (1 rejected / 5 accepted). Consider widening bounds or tightening the prior.
generated 5 prior draws from model IR (7 sampled params; 0 other parameter(s) resolve per cell from --params, the scenario's set or --param)
```

Combining an explicit draws **file** with a scenario that sets or scales one of
its columns is refused, because the scenario would override that column in every
draw:

```
error: parameter `mu` is controlled by both the draws file `draws.tsv` and scenario `pin_mu`
  draws file:  mu = [0.3, 0.4]
  scenario:    set mu = 0.3
The scenario would override the file's `mu` in every draw, so that column would have no effect.
Fix: drop the `mu` column from the draws file, or use a scenario that does not touch it.
```

The same collision with **generated** draws (`--draws prior|uniform|posterior`)
is not an error: the scenario wins for that parameter, so every draw resolves to
the same value of it, and the replacement is reported on stderr (§1.3, §5.4).

### 5.2 CLI ↔ Type Mapping

`camdl simulate` parses into `args::SimulateArgs` (clap) and lowers to
`sim_job::SimulateJob`; `camdl predict` builds the same job for its free-forward
replay. From there neither the engine nor the store knows which verb produced
the job.

```rust
/// The resolved engine input.
pub struct SimulateJob {
    pub model: String,                 // resolved IR or .camdl path
    pub params_files: Vec<String>,     // --params, applied in order
    pub backend: ForwardBackend,       // gillespie | chain_binomial | ode
    pub dt: f64,
    pub integrator: Option<IntegratorArg>,   // --integrator rk4|rk45
    pub source: ParamSource,           // where parameter vectors come from
    pub scenarios: Vec<ScenarioRef>,   // σ layer; empty ⇒ the implicit `(none)` arm
    pub t_end_override: Option<f64>,   // --to, resolved to model time
    pub required_output_times: Vec<f64>, // extra snapshot times (camdl predict)
    pub init_state: Option<Arc<InitStateSource>>, // --init-state ensemble
    pub obs_anchors: Option<ObsAnchorTimes>, // resolved first_obs/last_obs, once per job
    pub seeds: Seeds,                  // S layer
    pub cli_overrides: Vec<(String, f64)>,   // --param (highest M tier)
    pub fit_fixed: IndexMap<String, f64>,    // --draws prior|<file> --fit: the [fixed] block
    pub set_vec_entries: Vec<(String, String)>, // --param-vec
    pub table_files: Vec<(String, String)>,  // --table
    pub obs: ObsOutput,
    pub parallel: usize,               // rayon threads; --parallel (absent 1, 0 = all cores)
}

pub enum ParamSource {
    /// One point, run `replicates` times. `simulate` with no --draws.
    Point { replicates: usize },
    /// Design points: --sweep (name-sorted Cartesian product) or --sweep-file rows.
    Sweep { points: Vec<SweepPoint>, replicates: usize },
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
    /// An inline patch (the ad-hoc arm: --enable/--disable/--param).
    Inline { name: String, enable: Vec<String>, disable: Vec<String>,
             params: IndexMap<String, f64> },
}

pub enum Seeds {
    /// One base seed; replicates derive seeds by XOR-mixing.
    Single(u64),
    /// An explicit list (`--seeds`), used verbatim.
    Explicit(Vec<u64>),
}

pub enum ObsOutput { None, File(PathBuf), Dir(PathBuf), OnlyFile(PathBuf), OnlyDir(PathBuf) }
```

`--seeds` accepts an inclusive range `1:100` or a comma list `1,2,42`.
`--scenario` accepts a comma-separated list or repetition; both expand to the
same `Vec<ScenarioRef>`.

`simulate` hands each cell the store has not got to the engine, and renders its
exports from the stored cells afterwards.

### 5.3 Total Runs Calculation

```
Total runs = |param_points| × |scenarios| × |seed slots|

where |param_points| =
  Point:   1
  Sweep:   the number of design points (the name-sorted Cartesian product of
           the --sweep grids, or the rows of --sweep-file)
  Draws:   number of draw rows

and |seed slots| =
  --seeds given:   the explicit seed count
  otherwise:                       `replicates` (default 1), seeds derived from --seed

and |scenarios| =
  none specified:  1 (the implicit arm, `(none)`)
  otherwise:       the number of --scenario names
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

This matters when a preset that pins parameters is selected — the implicit arm
of a run with no `--scenario` applies no preset, but `--scenario baseline` on a
model whose `baseline` preset sets parameters applies those values over every
draw. The replacement is total for the parameter but reported (§1.3): one
`info:` line per replaced parameter, with the range of discarded values.

```
info: scenario `lockdown` replaced I0 = 77.09 to 875.81 (from the draw or sweep point) with 3, across 3 cells
```

The effect on the output is easy to miss without that line:

```bash
# sir_basic.camdl's `baseline` preset sets beta, gamma, N0, I0.
camdl simulate sir_basic.camdl --scenario baseline --draws uniform -n 3 --backend ode
```

produces three leaves with three distinct `run_id`s and three **byte-identical**
trajectories. Dropping `--scenario baseline` gives three different trajectories.

For user-authored coordinates this cannot happen: a `--sweep` grid, a
`--sweep-file`, or a `--draws <file>` whose parameters the effective scenario
sets or scales is refused before any cell runs (§3.2, §3.6). Generated draws
stay allowed, because a scenario over them is the intended counterfactual, so
for them the rule remains practical: when drawing a parameter, the scenario in
force must not `set` it unless discarding the draws for that parameter is what
you mean. Compare two cells' `traj.tsv` to check.

### 5.5 The design axis: `--sweep` and `--sweep-file`

A _design point_ is a parameter value the modeller chose; a _draw_ is a sample
from a measure. `simulate` keeps the two apart: `--draws` builds the sample axis
and pools its cells into a band, `--sweep` / `--sweep-file` build the design
axis and keep every point distinct. The two are refused together.

**`--sweep PARAM=GRID`**, repeatable. GRID is a comma list (`beta=0.2,0.3,0.6`),
`lin(min,max,n)` or `log10(min,max,n)`; an indexed parameter is named
underscore-joined (`beta_north`), and its column is `sweep:beta_north`. Several
`--sweep` flags form the Cartesian product **sorted by parameter name**, the
first name varying slowest; `point_id` is a point's 0-based position in that
product, and `camdl predict --sweep` gives the same `point_id` for the same
flags. A parameter named twice, or one the model does not declare, is refused
before anything runs.

**`--sweep-file FILE`**: a TSV whose header names model parameters and whose
rows are design points, `point_id` the 0-based row. A `point_id` column of
distinct non-negative integers gives each row the id it is written under instead
(in the combined and observation tables); the row position still drives nothing
but the engine's own order. A column naming no model parameter is refused with
the parameter list; a file with zero rows is refused; a duplicate row is refused
naming both lines; a repeated or non-integer `point_id` is refused naming the
row. `--sweep-file` and `--sweep` are mutually exclusive. The file enters no
identity: each row's values reach the cells' parameters.

**`camdl design lhs MODEL --fit FIT.toml --n N [--seed S] [-o FILE]`** writes
such a file: a Latin-hypercube design of N points over the fit config's
`[estimate]` parameters, on stdout or to FILE. The header is `point_id`, then
the parameters in name order; `point_id` is the 0-based row; each value is on
the natural scale, written in the shortest decimal form that reads back to the
same floating-point number. Each parameter is sampled within its search box, the
entry's `bounds` or else the range the model declares (§6.3). The box is cut
into N bins of equal width, on the log scale for a positive-kind parameter
(`rate`, `positive`, `count`, `duration`) whose lower bound is above zero and on
the linear scale otherwise, and each bin holds one point. The design is a
function of the model, the fit config, N and the seed (default 1): the same four
give the same bytes, and the command writes nothing to the store. `--n` below 2
is refused; so is a parameter whose search box has an infinite end, named with
its box, since such a box has no bins of equal width. `--fit` takes a fit config
or a stored fit's handle; the model sampled is MODEL, and the config's `[model]`
is not read.

**Seeds and common random numbers.** Every design point at a given replicate
shares one process seed: with `--replicates 2`, point 0 and point 1 of replicate
0 run on the same random stream, as do their replicate-1 cells. So a parameter
the dynamics do not read gives byte-identical trajectories across points, paired
differences across points are low-variance, and adding a point leaves the
others' trajectories unchanged. For independent noise per point, run one
invocation per point with a distinct `--seed`. The pairing key is the `seed`
column of the combined and observation tables.

**Refusals.** A scenario whose `set` or `scale` touches a swept parameter —
directly or through a preset it composes — is refused (§3.6), on the run and the
dry run alike; so is a `--param` on a swept parameter, since `--param` outranks
every tier and the sweep would not vary it; so is `--sweep` with `--draws` or
`--design-from`.

**Labels and records.** Each cell's parameters group is labelled by the swept
values, `beta=0.2` in its labels and `beta_0.2-<12 hex>` in the path, two swept
parameters as `beta=0.3_rho=0.5`; the arm label is the preset's name, or derived
from the flags for an ad-hoc arm (`(none)`, `(none)+vaccination` for
`--enable vaccination`, `(none);beta=0.3` for `--param beta=0.3`). The
invocation's record lists the cells in order with their labels, which carry each
cell's `point_id` and swept values.

**Columns.** Under a sweep the combined table (`-o`, `--stdout`,
`camdl cat @LABEL`) leads with `replicate`, `scenario`, `point_id`, one
`sweep:<param>` column per swept parameter (shortest round-trip rendering, so
`0.1234567` reads back exactly) and `seed`, then the time columns; there is no
`draw` column. Observation tables under `--obs-dir` / `--obs-only-dir` / `--obs`
/ `--obs-only` carry the same leading columns before the stream's own.
`quantities {}` under a sweep are banded per (design point, scenario), never
across points; one replicate per point gives a point-mode row per group.

**Observations as a dataset.** `--obs-dir DIR` writes one TSV per observation
stream, stacked over every cell with the columns above; that directory is the
dataset a modeller then hands to `pfilter --data cases=DIR/cases.tsv` or to a
fit, and it is rendered from the `obs` by-product stored in each cell. There is
no path-less observation flag: the destination is the product.

```bash
camdl simulate ocaml/golden/sir_basic.camdl --params ocaml/golden/sir_basic.params.toml \
    --sweep beta=0.2,0.5 --replicates 2 --dry-run
```

```
2 design points × 2 replicates = 4 runs
camdl simulate (dry run)

  model: ocaml/golden/sir_basic.camdl
  backend: chain_binomial
  dt: 1
  sweep: beta = [0.2, 0.5]
  design: 2 design point(s)
  seed: 1
  replicates: 2
  scenario: (none)

Parameters (3):
  I0    =             10  ocaml/golden/sir_basic.params.toml
  N0    =           1000  ocaml/golden/sir_basic.params.toml
  gamma =       0.100000  ocaml/golden/sir_basic.params.toml

  total: 2 design points × 2 replicates = 4 runs
info: the draw or sweep point replaced beta = 0.3 (from ocaml/golden/sir_basic.params.toml) with 0.2 to 0.5, across 4 cells

store: camdl-store
  [miss] scenario=(none) seed_1
  [miss] scenario=(none) seed_1-r1
  [miss] scenario=(none) seed_1
  [miss] scenario=(none) seed_1-r1
  0 cache hits → read back, 4 misses → would simulate
(dry run — no simulation, no files written.)
```

Each cell is named by its scenario and seed label; once the cells are stored,
each line reads `[hit ]` and ends with the cell's 12-hex id. Both `beta` points
run `seed_1` and `seed_1-r1` — the common random numbers above — and `beta` is
absent from the parameter list because the sweep supplies it per cell.

#### Scoring a design: `pfilter --sweep-file`

`camdl pfilter --sweep-file FILE` estimates the log-likelihood log p(y | θ) at
every row of a design file, one table row per design point. With a space-filling
design from `camdl design lhs`, the table is a likelihood landscape
([Commands](commands.md#evaluate-the-model-over-a-parameter-design)).

```bash
camdl pfilter sir.camdl --fit fit.toml --sweep-file design.tsv \
    --particles 1000 --replicates 5 --seed 1 -o landscape.tsv
```

The file is read as `simulate --sweep-file` reads it: its header names model
parameters, an optional `point_id` column names the rows, and the same files are
refused. Each design point is scored exactly as the single call
`camdl pfilter sir.camdl --fit fit.toml --param beta=… --param gamma=… --particles 1000 --replicates 5 --seed 1`
would score it:

- **Parameters.** A point's values enter at tier 3.5 (§1.3). Every other tier
  resolves as it does for a single call: the fit config's `[fixed]`, `--params`,
  the scenario.
- **Seeds.** A point does not enter the seed. Every point runs its
  `--replicates` filters on the seeds of the single call with the same `--seed`,
  so the filters at different points share random numbers, and the errors of the
  rows are correlated.
- **The store.** Each point is stored as the single call's `Pfilter` result,
  with that call's `run_id`. A point already stored, by a single call or by
  another design, is reported `cached` and not recomputed, so an interrupted
  design run again computes only the remaining points. The design file's digest
  enters no identity, and the table itself is not stored: it is rendered from
  the stored points, so `-o` and `camdl cat @L` of a design run with `--label L`
  write the same table.

Points run concurrently on the `--parallel` thread pool. The table does not
depend on the schedule.

**The table**, to `-o PATH` or stdout, one row per design point in file order:

| column          | meaning                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `point_id`      | the row's `point_id`, else its 0-based row                                                        |
| `sweep:<param>` | one per design column, name-sorted: the value scored                                              |
| `loglik`        | the mean over replicates of the estimated log-likelihood; `-inf` when any replicate failed        |
| `loglik_sd`     | the standard deviation over replicates; empty under `--replicates 1` or when any replicate failed |
| `n_failed`      | how many replicates failed (below)                                                                |
| `failure`       | empty when every replicate scored; else the first failed replicate's message, on one line         |
| `run_id`        | the stored result (`camdl cat <run_id>`); empty under `--no-store`, whose store is deleted        |

Numbers are written in shortest round-trip form, so a `loglik` read back equals
the stored one.

**Failed points.** A replicate fails at a design point in either of two ways: it
scores `-inf` (every particle's weight is zero at some observation), or the
filter stops with an error, such as a degeneracy error when the effective sample
size stays collapsed over several observations. Both are recorded in the row and
neither stops the run. In the `failure` column the message is on one line: tabs
and newlines are replaced by spaces, and a degeneracy error leaves out the
wall-clock time it carries, so a rerun writes the same table. Either failure is
determined by the point, so the point is stored as a completed result holding no
log-likelihood, the number of replicates that failed and the first one's
message, and a rerun reuses it. A point whose every replicate failed has
`loglik = -inf` and `n_failed` equal to `--replicates`. After the table, the run
prints how many points failed entirely and the first failure message. The run
exits with status 3, the status of a single call whose filter failed, only when
every point failed, and with 0 otherwise. A single call without `--sweep-file`
whose filter fails at every replicate is stored the same way and exits with
status 3.

A point that does not complete for a reason a rerun could change, such as the
compute budget of `--pf-max-substeps` or a store or I/O error, is not stored.
The run names each such point, writes no table and exits with status 1; a rerun
computes only those points.

**Refused beside `--sweep-file`**, before any filter runs:

- `--param` on a design column, and a `--scenario` that sets or scales one:
  either would score every row at one value while its `sweep:<param>` column
  said otherwise;
- every output that describes one filter run: `--trace`, `--pf-health`,
  `--save-final-state`, `--save-paths`, `--save-filtering`,
  `--save-filtered-streams` and `--save-prequential`.

A parameter that no tier gives a value, such as an `[estimate]` parameter the
file has no column for, is refused as it is for a single call ("parameter 'beta'
has no value").

### 5.6 Execution Flow

A multi-cell `camdl simulate`:

1. Parses the flags, resolves the model to IR (compiling the `.camdl` if
   needed), loads `--params`, and resolves every `--scenario` against the
   model's presets (or derives the ad-hoc arm from `--enable` / `--disable` /
   `--param`).
2. Expands the parameter axis — the draws of `--draws`, or the design points of
   `--sweep` / `--sweep-file` — and builds the `param-point × scenario × seed`
   grid. A scenario touching a user-authored coordinate is refused here.
3. Resolves each cell's identity (§3.7). `--dry-run` stops here and prints the
   plan, each cell as a hit or a miss, and the counts.
4. Stores the compiled model, then begins every cell in the store: a stored cell
   is reused, and each other cell is simulated, in parallel under `--parallel`,
   and stored with the by-products asked of it. Each cell's seed is derived from
   its coordinates, so no ordering can perturb a trajectory.
5. Records the invocation: the inputs as given, the `--label`, and the cells in
   order with their labels.
6. Lists every stored failure and exits 1 if there is one (§4.4); otherwise
   reports how many cells were read from the store and where the cells landed,
   and renders the `-o` / `--stdout` / `--obs*` / `--quantities-out` exports
   from the stored cells.

## 6. FitConfig — the inference type

### 6.1 Overview

A `fit.toml` is one (problem, method) pair: which model to fit, what data to fit
it to, which parameters to estimate versus hold fixed — the _problem_, which
every command that takes `--fit` reads — and one `[method]`, the algorithm that
fits it, which only `fit run` reads. It defines a _view_ of the parameter space
— the partition of the model's parameter set into free parameters (explored by
the algorithm) and fixed parameters (held constant). The algorithm then operates
in the reduced space of free parameters. (See Buffalo 2026 for the formal
treatment of parameter views, transforms, and the downward chain from inference
coordinates to simulator output.)

The runtime type is `FitConfig` (`rust/crates/cli/src/fit/config_v2.rs`), a
`Problem` and an `Inference` split from one flat parse. `Problem::load` is the
entry point for every non-fit reader (`simulate --draws prior --fit`,
`pfilter --fit`, `profile --fit`) and discards the inference half, so a file
with no `[method]` is a complete problem for them; `camdl fit
run` — the sole
entry point that executes a fit — reads both and refuses a file with no
`[method]` by name.

Three properties govern the whole surface:

- **Strict parsing.** The flat wire struct and every nested config struct carry
  `#[serde(deny_unknown_fields)]`, so a misplaced or misspelled key is a load
  error, not a silent drop. The one exception is `[fixed]`, whose
  `#[serde(flatten)]` map of arbitrary `param = value` entries is incompatible
  with the attribute. The `[method]` table needs a separate mechanism — see
  §6.9.
- **Paths in the file anchor at the file.** `[model] camdl`, `store`,
  `[data] file`, `[data.observations]`, and `[data.holdout]` are resolved
  against the `fit.toml`'s own directory at load time (`anchor_paths_at`,
  `config_v2.rs`), so a config is relocatable as a unit and runs the same from
  any working directory. Two path-bearing keys are **not** in that set and
  anchor at the process working directory instead: `[fixed]
  from_file` and
  `[synthetic] true_params`.
- **Absolute paths warn.** `absolute_path_warnings` (`config_v2.rs`) emits one
  stderr line per absolute reference in `[model] camdl`, `store`, `[data] file`,
  `[data.observations]` or `[data.tables]`, because an absolute path pins the
  config to one machine's layout. It is a warning, not an error.

### 6.2 Top-level keys

Every key a `fit.toml` accepts at the top level. Anything else fails at load
with the serde list quoted in §6.9.

| key              | type            | required  | default                | meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------- | --------------- | --------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `[model]`        | table           | **yes**   | —                      | `camdl = "<path>"`. Accepts a `.camdl` source or a pre-compiled `.ir.json`.                                                                                                                                                                                                                                                                                                                                                                                        |
| `[data]`         | table           | one of    | —                      | Real-data source. Mutually exclusive with `[synthetic]`; exactly one must be present.                                                                                                                                                                                                                                                                                                                                                                              |
| `[synthetic]`    | table           | one of    | —                      | Generate N datasets from known truth and fit each (simulation-based calibration).                                                                                                                                                                                                                                                                                                                                                                                  |
| `[estimate]`     | table of tables | **yes**   | —                      | The free parameters. See §6.3.                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `[fixed]`        | table           | **yes**   | —                      | The held-constant parameters. See §6.4.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `[method]`       | table           | `fit run` | —                      | One algorithm, its settings, and the chains' `starts`. Absent, the file is a complete problem for every other reader and `fit run` refuses it. See §6.5.                                                                                                                                                                                                                                                                                                           |
| `[config]`       | table           | no        | `{ dt = 1.0 }`         | Fit-wide simulator settings. See the sub-table below.                                                                                                                                                                                                                                                                                                                                                                                                              |
| `store`          | string          | no        | `camdl-store`          | The run store's root for every command that reads this file, below `--store` and above `CAMDL_STORE` (§2.1). Anchored at the `fit.toml`. Not part of any identity.                                                                                                                                                                                                                                                                                                 |
| `fit_seeds`      | list of ints    | no        | `[--seed]` (CLI, or 1) | One fit per listed seed. Duplicates rejected.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `simplex_groups` | array of tables | no        | `[]`                   | `[[simplex_groups]] params = ["a","b",…]` — members must form a probability simplex. Honored by IF2 only; other algorithms warn.                                                                                                                                                                                                                                                                                                                                   |
| `scenario`       | string          | no        | none                   | Named scenario from the model; applies its enable/disable lists and param overrides before inference. Exclusive with `enable`/`disable`.                                                                                                                                                                                                                                                                                                                           |
| `enable`         | list of strings | no        | `[]`                   | Ad-hoc intervention enable list; `"*"` enables every toggleable intervention.                                                                                                                                                                                                                                                                                                                                                                                      |
| `disable`        | list of strings | no        | `[]`                   | Ad-hoc disable list. Explicit disable beats `always_active`.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `ic_free`        | bool            | no        | `false`                | Condition the likelihood on `y₁` rather than on a committed initial state. Requires particles that differ in x₀: `if2` always, `pfilter`/`pmmh` only when the model's `init { }` declares a law (gh#732). Also requires a non-missing `y₁`, and — when the model's `init { }` is deterministic — an `if2` method with a `[method.perturbation]` entry that sets `initial_state = true`. `profile --fit` and `pfilter --fit` score the same conditional likelihood. |
| `[provenance]`   | table           | no        | none                   | Lineage metadata. See §6.7.                                                                                                                                                                                                                                                                                                                                                                                                                                        |

`[config]` (`FitBackendConfig`, `config_v2.rs`):

| key                      | type  | default                      | meaning                                                                                                                                                                 |
| ------------------------ | ----- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dt`                     | float | the model's `dt`, else `1.0` | Integrator step. A model's declared `simulate { dt }` is the step; a `[config] dt` that differs is refused, an agreeing one accepted. A top-level `dt` is a load error. |
| `allow_degenerate_rates` | bool  | `false`                      | Treat a numerical collapse in a rate expression (div-by-zero, `Pow`→NaN, `Sqrt` of a negative) as `0.0`.                                                                |

The synthetic-data backend is `[synthetic] backend`. Any other key under
`[config]` is refused at load as an unknown field.

`[data]` (`DataSpec`, `config_v2.rs`):

| key             | type            | meaning                                                                                                                                                                                 |
| --------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `file`          | string          | Single wide TSV holding one column per model-declared observation stream. Exclusive with `observations`.                                                                                |
| `observations`  | map name → path | Per-stream file paths, keyed on the `observations { }` block names in the `.camdl`. Exclusive with `file`.                                                                              |
| `holdout_after` | time spec       | Temporal split: observation rows whose end is after this value are removed from training (see below). Exclusive with `holdout`.                                                         |
| `holdout`       | map name → path | Explicit held-out files, tail-only; the training streams stay whole, and each held-out row's date, not its value, enters the problem's identity (§2.2). Exclusive with `holdout_after`. |

Exactly one of `file` / `observations` must be set. The observation model
(likelihood family) and projection (which flow or compartment is accumulated)
are declared in the `.camdl`; `fit.toml` supplies only the file paths.

A holdout declaration is applied when the fit loads its data
(`apply_holdout_declaration`, `fit/runner.rs`), which decides the fit's training
partition (`fit/training_partition.rs`). `profile --fit` applies it to its bound
data through the same function; the training rows a cutoff leaves each stream
enter the profile's base identity, omitted without one. `pfilter` sees the full
series, and a stored fit's held-out rows are scored from outcome files by
`camdl score` and `camdl compare` (Appendix A, `camdl compare`). The one
exception is `pfilter --save-final-state` under `holdout_after`: it binds only
the training rows, cut by the training rule below (`cutoff_prefix`,
`fit/training_partition.rs`), and saves the state at their end, since a state
filtered through held-out rows would be a forecast origin that has seen the data
its forecast is scored against (gh#1138). Every per-run output of that run then
covers the training rows only: the log-likelihood, `--trace`,
`--save-prequential`, `--save-paths`, `--save-filtering`,
`--save-filtered-streams` and `--pf-health`. `--score-from` is refused beside a
cut that dropped rows, since every row it could score trained. When the cut
drops a row, the training rows it kept enter the run's `config` level; a cut
that drops nothing keeps the run_id of the same run without
`--save-final-state`. An anchored cutoff resolves over the fit config's own
files, so a run that binds only the training rows with `--data` cuts nothing and
saves the same state.

- `holdout_after` is a bare model time, a date (the instant that day begins), or
  an observation-anchored expression (`last_obs - 6 weeks`). Its anchors resolve
  over the full files: `first_obs` is the earliest start and `last_obs` the
  latest end of a row with a value, across every bound stream; an `NA` row
  counts toward neither. A row trains if the end of what it measures (its
  instant, or its period's end) is at or before the cutoff, so each stream
  trains a prefix of its rows. Every bound stream must keep a training row with
  a value and some row with a value must be held out; either failing is a hard
  error. An `NA` row keeps its position in the prefix and in its row digest, but
  counts toward neither check.
- `[data.holdout]` names a stream and a file of held-out rows. Every row of the
  training files trains, and every held-out row must end strictly after the
  training extent, the latest end among the training rows of every bound stream.
  An interleaved holdout is refused; only tail holdouts are supported.

The fit prints the rows it trained on, the training extent and what was withheld
(for example
`holdout: trained on 9 row(s) through t = 10; 7
observation row(s) ending after t = 10 withheld from training`,
followed by a note that `camdl compare` scores them out of sample), and the
first held-out row of each stream as the file writes it, and records the
partition in `fit.meta.json` (`training_partition`). `camdl score` scores such a
fit's held-out rows from an outcome file whose leading rows equal its training
rows by row digest (§2.2.4), and `camdl compare` compares those rows, stamped
`held-out tail` with `train_end` the training extent; a fit without a holdout is
compared on its training rows, in-sample.

A model anchor (`simulate { to = last_obs + … }`, a `piecewise` breakpoint) and
`camdl predict`'s `value_at(…, last_obs)` resolve over the training rows only,
so appending held-out rows to a `holdout_after` file moves no anchor.

`[synthetic]` (`SyntheticSpec`, `config_v2.rs`):

| key           | type                               | required | default          | meaning                                                                              |
| ------------- | ---------------------------------- | -------- | ---------------- | ------------------------------------------------------------------------------------ |
| `true_params` | string                             | **yes**  | —                | Flat TOML of `name = value` ground truth. Anchored at the working directory.         |
| `sim_seeds`   | `"N:M"` range **or** list of ints  | **yes**  | —                | One dataset per seed. Duplicates rejected; a malformed range (`"1-20"`) errors.      |
| `datasets`    | int                                | no       | `len(sim_seeds)` | Must equal `len(sim_seeds)` when supplied.                                           |
| `scenario`    | string                             | no       | none             | Scenario applied during **generation** only, not during fitting.                     |
| `backend`     | `chain_binomial`/`gillespie`/`ode` | no       | `chain_binomial` | Forward backend used to generate the datasets. Distinct from the method's `backend`. |

**Where scoring begins** is not a `fit.toml` concern. Every incidence stream
declares what each row covers (`covers = ...`, or `window_start`/`window_stop`
columns — `camdl-data-spec.md`, "What a row covers"), and a declared first
period opens where it says: the span from `t_start` to that opening is simulated
and scored nowhere, and the first datum is scored against its own period. A
first period opening before `t_start` is a load error naming both times.
`camdl pfilter`, `camdl profile` and `camdl predict` bind the same declared
periods, so a fixed-θ loglik and a predictive row cover the window the fit
scored.

### 6.3 `[estimate]` — free parameters

Each entry is `[estimate.<param>]` (or an inline table under `[estimate]`).
`EstimateSpecV2` (`config_v2.rs`):

| key      | type         | default                     | meaning                                                                                                                                                                                                                                                                                                                                                                                 |
| -------- | ------------ | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bounds` | `[lo, hi]`   | the model's `in [lo, hi]`   | Search box. May only **narrow** the model's declared range; loosening is an error.                                                                                                                                                                                                                                                                                                      |
| `start`  | float        | a draw within the bounds    | The base point: where `starts = "single"` puts every chain, and where `uniform` puts chain 1; the other spread rules read it only at one chain, and the sourced rules not at all. Under `pmmh`/`mh` it is also the point of the pre-run likelihood check. A written `start` is θ₀, the point at which the launch table and `run_inputs.json` give each step's natural-scale equivalent. |
| `prior`  | inline table | the model's `~` declaration | See the wire format below. Required in some form for `pgas`/`pmmh`/`mh`.                                                                                                                                                                                                                                                                                                                |

An entry with no fields at all (`beta = {}`) is legal: it means "estimate this,
take the bounds and the prior from the model." A step is not an `[estimate]`
key: IF2's perturbation is `[method.perturbation]` and the initial proposal SD
of `pmmh` and `mh` is `[method.proposal]` (§6.5). An entry that writes `rw_sd`
or `perturb_only_at_t0` is refused before the typed parse, and the message
gives, per parameter, the entry to write in the method's table: the old
natural-scale step converted to z at θ₀ (its `start`, else the model's value,
else the middle of its box; under `pmmh` and `mh` five times that, the old
initial proposal), with the natural-scale step the default would give there.
`fit-toml.md`, "Step sizes", gives the conversion.

**Transforms** (`runner.rs`, `derive_transform_with_bounds`;
`sim::inference::types::Transform::for_domain`) map the real line onto the
_resolved_ bounds, chosen by which ends are finite:

| resolved bounds       | declared type                           | transform (record tag)                         |
| --------------------- | --------------------------------------- | ---------------------------------------------- |
| `(−∞, ∞)`             | any                                     | θ = z (`identity`)                             |
| `[lo, ∞)`             | any                                     | θ = lo + eᶻ (`lower_bounded`)                  |
| `(−∞, hi]`            | any                                     | θ = hi − eᶻ (`upper_bounded`)                  |
| `[lo, hi]`, 0 < lo    | `rate`, `positive`, `count`, `duration` | ln θ = ln lo + ln(hi/lo)·σ(z) (`log_interval`) |
| `[lo, hi]`, otherwise | any                                     | θ = lo + (hi − lo)·σ(z) (`interval`)           |

No transform clamps, and every finite end is z = ±∞, so narrowing in `fit.toml`
genuinely narrows the IF2 search and the samplers' target. Resolution order is
`fit.toml [estimate].bounds` → the model's `in [lo, hi]` → the range the
declared type carries: `[0, 1]` for `probability`; `(−∞, ∞)` for `real` and
`instant`; `[0, ∞)` for `rate`, `positive`, `count` and `duration`. A parameter
whose IR records no type is refused. The record (`fit.meta.json` `partition`)
shows an unbounded `real` as `search: ["-inf", "inf"]`.

**Prior wire format.** Externally tagged inline tables, matching what the OCaml
compiler emits for in-model `~` priors (`PriorDist`,
`rust/crates/ir/src/parameter.rs`). Field names are exact — `normal` takes
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

**Prior precedence** (three tiers, `validate_priors_present`, `config_v2.rs`): a
`fit.toml` `prior` wins over the model's `~` declaration; an explicit
`{ flat = {} }` counts as declared. A sampler method whose estimated parameter
has none of the three is a **hard error** — `fit run` never falls back to flat
silently, because downstream consumers treat the chain as the canonical
posterior. (`camdl profile` falls back to flat without a warning, recording
`flat_fallback`; every profile cell is a maximization that reads no prior.)

**Prior support** (`validate_prior_transform_compat`) is enforced before any
fitting: the resolved bounds must lie within the prior family's support —
`[0, ∞)` for `log_normal`, `half_normal`, `gamma` and `exponential`, `[0, 1]`
for `beta`, `[lower, upper]` for `uniform` and `log_uniform`; `normal` and
`flat` accept any bounds, and `truncated_normal`'s truncation must equal them.
Every prior reads ln θ and ln(1 − θ) from the transformed coordinate, so the
density is exact under every transform.

### 6.4 `[fixed]` — held-constant parameters

`FixedParams` (`config_v2.rs`). Three sources, and the union of `[estimate]` and
`[fixed]` must be exactly the model's parameter set:

| key             | type   | meaning                                                                                            |
| --------------- | ------ | -------------------------------------------------------------------------------------------------- |
| `from_file`     | string | Bulk-load a flat TOML of `name = value`. Resolved against the **working directory**, not the file. |
| `from_scenario` | string | Bulk-load a named `scenario` block from the `.camdl`, following its `compose = [...]` chain.       |
| `<param> = <n>` | float  | Inline values. Override `from_file` entries on collision.                                          |

`from_scenario` is mutually exclusive with both `from_file` and inline values:
allowing an override would make the fit a hybrid that corresponds to nothing in
the `.camdl`. It imports the values `camdl simulate --scenario` runs at: every
value the scenario sets, with the scenario's `scale` factors applied, through
composed presets. A scenario that scales a parameter it does not set is refused,
since the import has no value for the factor to multiply; set the parameter in
the scenario as well. A scenario that sets or scales a parameter in `[estimate]`
is refused (§1.3): to estimate a parameter a scenario sets, name a scenario that
does not set it.

### 6.5 `[method]` — the inference method

A file carries one `[method]`: one algorithm with its settings and the chains'
`starts` rule (§6.6). A second way of fitting the same problem is a second file
— `camdl fit new --from fit.toml fit-pgas.toml` — and the two share one fit
problem (§2.2). The table is tagged by `algorithm` (`Method`, `config_v2.rs`,
`Algorithm` with `#[serde(tag = "algorithm")]`) and carries an explicit
`backend`. Each algorithm is valid on exactly one backend; the pair is checked
at load against the `METHODS` registry (`rust/crates/cli/src/fit/methods.rs`),
which is the single source of truth for `camdl fit methods`, the runtime
banners, and the invalid-pair error.

| `algorithm` | `backend`        | status | category   | role                                                                                                  |
| ----------- | ---------------- | ------ | ---------- | ----------------------------------------------------------------------------------------------------- |
| `if2`       | `chain_binomial` | stable | inference  | Iterated filtering → MLE.                                                                             |
| `pgas`      | `chain_binomial` | stable | inference  | Particle Gibbs + NUTS on θ → posterior. Default Bayesian path.                                        |
| `pmmh`      | `chain_binomial` | stable | inference  | Pseudo-marginal MH → posterior. Degrades for T > 500.                                                 |
| `pfilter`   | `chain_binomial` | stable | diagnostic | Bootstrap particle filter at fixed θ → loglik, ESS, prequential.                                      |
| `nl-sbplx`  | `ode`            | beta   | inference  | NLopt Sbplx → deterministic MLE. Robust at bounds.                                                    |
| `nl-bobyqa` | `ode`            | beta   | inference  | NLopt BOBYQA → deterministic MLE. Faster, fails at bounds.                                            |
| `nl-lbfgs`  | `ode`            | beta   | inference  | NLopt L-BFGS with the forward-sensitivity gradient → deterministic MLE; needs a differentiable model. |
| `mh`        | `ode`            | beta   | inference  | Gradient-free MH on the ODE marginal likelihood → posterior.                                          |

The `ode`-backend samplers fit the deterministic marginal likelihood
`p(y | θ, ODE skeleton)`, not the stochastic `p(y | θ)` — a different
statistical object, appropriate for equilibrium or large-population models.
`gillespie` is a forward-simulation backend with no inference interface and is
rejected at parse.

Keys common to every method: `algorithm`, `backend`, `starts` (§6.6).

**Step tables.** `[method.perturbation]` (`if2`) and `[method.proposal]`
(`pmmh`, `mh`) hold each estimated parameter's step in z, the unconstrained
coordinate of its map (§6.3); the tables below give each one's form and
defaults. Either table under another method is refused at load, and an entry
that names a parameter `[estimate]` does not list is refused before the run. A
parameter on the `identity` map has no default step, and its step must be
written. `pgas`, `pfilter` and the `nl-*` optimizers take no per-parameter step.
Before any chain runs, `fit run` reads from the compiled model where each
estimated parameter enters (the initial state, the dynamics or an observation,
directly or through a binding, forcing or table) and refuses:

- under `if2`, an estimated parameter that enters only the initial state without
  `initial_state = true`, and `initial_state = true` on a parameter that enters
  the dynamics or an observation, naming where it enters; when the compiled
  model holds a reference that cannot be resolved, `initial_state` is refused on
  every parameter;
- under `if2` and the `nl-*` optimizers, an estimated parameter the likelihood
  does not depend on, including one that appears only in another parameter's
  prior;
- under `pmmh`, `mh` and `pgas`, an estimated parameter with no connection to
  the likelihood, directly or as a hyperparameter of one that has, and a flat
  prior. One with a proper prior of its own runs, with a warning that its
  marginal posterior equals its prior.

The guide to choosing a step is `fit-toml.md`, "Step sizes".

**`if2`** — `chains`, `particles`, `iterations`, `cooling` required.

| key                    | default | meaning                                                                                                                                                                                                                                                                                                                                                |
| ---------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cooling`              | —       | Fraction of the initial perturbation SD remaining at `cooling_target_iters`.                                                                                                                                                                                                                                                                           |
| `cooling_target_iters` | `50`    | pomp's `cooling.fraction.50` convention; decoupled from `iterations`.                                                                                                                                                                                                                                                                                  |
| `loglik_eval`          | table   | Clean re-scoring of candidate θ̂. `n_particles = 4000`, `n_replicates = 8`, `combine = "log_mean_exp"` (or `"mean"`).                                                                                                                                                                                                                                   |
| `gate`                 | table   | `a_thresh = 1.01` (chain-agreement Â ceiling), `decibans_thresh = 30.0` (inter-chain loglik spread floor).                                                                                                                                                                                                                                             |
| `dt_check`             | table   | Post-fit Richardson dt-convergence audit. `enabled = true`, `n_halvings = 2`; `n_particles`/`n_replicates`/`combine` inherit `loglik_eval`; `threshold_nats` defaults per backend (2.0 chain_binomial, 0.5 ode_rk4).                                                                                                                                   |
| `perturbation`         | table   | Per estimated parameter, its random-walk SD in z: `NAME = sd`, or `NAME = { sd = …, initial_state = true }` for a parameter that enters only the initial state, perturbed at t = 0 only. A name left out takes 1/20 on an `interval` or `log_interval` map and 1/5 on a one-sided map; a simplex member takes 1/6 on its group's log-ratio coordinate. |

`iterations` must be ≥ 1. IF2 has no extension dimension, so `--resume` does not
apply: its cooling schedule is determined by the total iteration count, and
restarting mid-schedule is statistically incoherent.

**`pgas`** — `chains`, `particles`, `sweeps` required.

| key                    | default | meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `burn_in`              | `2000`  | Discarded sweeps. Must be `< sweeps`.                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `thin`                 | `5`     | Retain every k-th post-burn-in sweep.                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `max_tree_depth`       | `10`    | NUTS tree-depth ceiling for the θ\|X update.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `trajectory_warmup`    | `0`     | CSMC-only sweeps before parameter updates begin.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `binomial`             | `btpe`  | Binomial sampler for propagation draws: `btpe`, or `btrs` (faster, exact, not bit-compatible). `camdl predict` draws with the sampler the fit ran (D78).                                                                                                                                                                                                                                                                                                               |
| `csmc_sweeps_per_nuts` | `1`     | CSMC trajectory updates per parameter update.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `n_trajectories`       | `200`   | Posterior trajectories written to disk. Output-shaping, but keyed.                                                                                                                                                                                                                                                                                                                                                                                                     |
| `dense_mass`           | `true`  | Full-covariance NUTS metric; `false` for diagonal.                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `use_nuts`             | `true`  | `false` falls back to MH-within-Gibbs for θ\|X.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `ancestor_sampling`    | `true`  | `false` runs plain particle Gibbs without the ancestor-sampling move — a valid kernel on its own, since ancestor sampling is a mixing addition and removing it leaves the invariant distribution intact. A diagnostic control, not a recommendation: run the method both ways to measure how much the ancestor move is buying on this model. If path renewal is unchanged with it off, the move is contributing nothing and the freeze is structural in the genealogy. |

**`pmmh`** — `chains`, `particles`, `iterations` required.

| key           | default | meaning                                                                                                          |
| ------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `burn_in`     | `5000`  | Must be `< iterations`.                                                                                          |
| `thin`        | `10`    |                                                                                                                  |
| `adapt`       | `true`  | Haario adaptive Metropolis on the proposal SDs.                                                                  |
| `adapt_start` | `300`   | MCMC step at which adaptation begins.                                                                            |
| `proposal`    | table   | Per estimated parameter, its initial proposal SD in z, used as written: `NAME = sd`. A name left out takes 0.25. |

**`mh`** — `chains`, `iterations` required. Reuses the PMMH chain and
adaptive-proposal machinery with the particle-filter likelihood swapped for the
deterministic ODE evaluation, so it carries no `particles`.

| key           | default | meaning                                                                                                                                                                           |
| ------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `burn_in`     | `5000`  | Must be `< iterations`.                                                                                                                                                           |
| `thin`        | `10`    |                                                                                                                                                                                   |
| `adapt`       | `true`  |                                                                                                                                                                                   |
| `adapt_start` | `300`   |                                                                                                                                                                                   |
| `proposal`    | table   | As under `pmmh`.                                                                                                                                                                  |
| `burnin_dt`   | none    | Coarse RK4 step for the unscored warm-up `[t_start, first_obs)`. `< dt` is refused and `= dt` is off; prevalence-scored streams, `rk4`, no `balance`/`events`. Identity-defining. |

**`pfilter`** — `particles` required. No `chains` (it is always one); `starts`
names the one point it scores — the base point, or the point a sourced rule such
as `{ from_mle = "@scout" }` names.

| key                  | default | meaning                                                                     |
| -------------------- | ------- | --------------------------------------------------------------------------- |
| `replicates`         | none    | Independent filter passes; the run reports loglik mean ± SD across them.    |
| `record_ancestry`    | `false` | Record per-step ancestor indices for smoothing-path reconstruction.         |
| `record_prequential` | `true`  | Record per-step predictive samples and log-likelihoods for `camdl compare`. |

**`nl-sbplx` / `nl-bobyqa` / `nl-lbfgs`** — `backend`, `chains` required
(`NloptConfig`, `config_v2.rs`). All three read the same keys.

| key         | default | meaning                                                              |
| ----------- | ------- | -------------------------------------------------------------------- |
| `tolerance` | `1e-6`  | NLopt `xtol_rel`.                                                    |
| `max_evals` | `5000`  | Per-chain objective-evaluation budget; hitting it is a soft failure. |
| `gate`      | table   | Two-leg convergence gate, same shape as IF2's.                       |

`chains` here is the number of independent multi-start optimizations;
`starts = "single"` defeats multi-start, since every chain then converges from
the same point.

`nl-sbplx` and `nl-bobyqa` are derivative-free: one objective evaluation is one
ODE solve. `nl-lbfgs` is the gradient method — one objective evaluation is one
_augmented_ solve carrying `∂x/∂θ` alongside the state, returning the
log-likelihood and its gradient together (the forward-sensitivity gradient,
`sim::inference::ode_grad::det_grad`). The budget and the progress step count
that unit for all three.

Because it takes a gradient, `nl-lbfgs` accepts only a **differentiable model**
— the gradient capability check refuses an unsupported rate or observation
gradient, an adaptive `rk45` integrator, a scheduled effect whose jump the
sensitivity cannot cross (a reactive intervention, a fire time that reads an
estimated parameter, a `transfer(count = …)`, an amount that reads compartment
state), or an initial condition the gradient path cannot seed. Fixed-time
`transfer(fraction = …)`, `add` and `set` effects, with constant or estimated
amounts, are carried: the step is split at the effect instant and the
sensitivity jumps with the state (`camdl docs inference`, "The differentiability
requirement"). It is checked at config validation, before any leaf is claimed,
and the refusal names `nl-sbplx` as the derivative-free alternative on the same
likelihood. If the gradient later turns out not to be computable at some θ the
search reaches, that chain fails with the reason and the θ rather than scoring
the point badly and continuing — L-BFGS has no direction to search in there.

**Observation periods and the step grid.** Where a method can place an observed
period's start and end is a capability of the method, recorded in the `METHODS`
registry (`PeriodBoundaries`, `methods.rs`), not a setting. `if2`, `pfilter` and
`pmmh` step exactly to every start and end, with a shortened step where one
falls between grid times. `pgas` steps only on its grid `t_start + k·dt`: once
the method is known, `FitRunConfig::build` checks every row's period, held-out
rows included, measured from `t_start`, and refuses the run if any start or end
lies off the grid (within `schedule::GRID_TOL`), naming the stream, the row, its
period and the two grid times either side
(`runner::refuse_periods_off_the_grid`). The ODE algorithms read the trajectory
at its output times and refuse a boundary with no output time. No method moves a
period, and no setting chooses how a period relates to the grid.

**IC-free support.** `ic_free = true` is honored by `if2`, and by `pfilter` /
`pmmh` on a model whose `init { }` declares a law
(`methods.rs::validate_ic_free`). `pgas` and the ODE algorithms score every
observation unconditionally, so the combination is rejected rather than allowed
to compute the unconditional likelihood under a banner claiming otherwise.
`pfilter` / `pmmh` on a _deterministic_ `init { }` are rejected for the other
reason: their bootstrap filter draws x₀ per particle, but every such draw
returns the same state, so the first reweight has nothing to discriminate
between (gh#732).

### 6.6 Chain starts: `starts`

One key on `[method]` answers where the chains begin. Its wire form is a bare
string for a rule that needs no source, or a one-key table naming the source
(`ChainStarts`, `rust/crates/cli/src/fit/starts.rs`):

```toml
starts = "from_prior"
starts = { from_posterior = "@scout" }
```

A rule is one of two kinds, and the kind is what the diagnostics act on. A
**spread** rule gives every chain its own independent point, so a between-chain
R̂ over the chains is a verdict; a **point** rule puts every chain at one point,
so the chains agree by construction and R̂ cannot say whether the posterior was
explored.

| `starts`                          | kind   | where the chains start                                                                                                                                            |
| --------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `from_prior`                      | spread | One draw per chain from each parameter's resolved prior (`fit.toml` `prior` over the model's `~`, §12.1). The default whenever every estimated parameter has one. |
| `uniform_unconstrained`           | spread | i.i.d. `U(-2, 2)` on the unconstrained scale, squashed and mapped into bounds. Boundary-avoiding and scale-invariant; Stan's default. The fallback default.       |
| `lhs`                             | spread | Latin-hypercube stratified over bounds, log-scale-aware for `Log` parameters. Best full-bounds coverage at low chain counts.                                      |
| `uniform`                         | spread | Per-chain uniform draw within natural-scale bounds. Clumps for `Log` parameters.                                                                                  |
| `{ from_posterior = "<handle>" }` | spread | One row per chain, drawn uniformly with replacement from a fit's `draws.tsv` (or a draws TSV path). Refused on a source that wrote no `draws.tsv`.                |
| `single`                          | point  | Every chain at the base point (`[estimate].start`, else the model's value).                                                                                       |
| `{ from_mle = "<handle>" }`       | point  | Every chain at a stored fit's estimate (its leaf's `fit_state.toml`).                                                                                             |
| `{ from_params = "<path>" }`      | point  | Every chain at the point in a hand-written flat params TOML.                                                                                                      |

`uniform`, `lhs`, and `uniform_unconstrained` collapse to the base point at
`chains = 1`; the source-reading rules do not. At two or more chains they refuse
a parameter with no finite box to draw within (an unbounded `real` or `instant`
on `(−∞, ∞)`, or an unbounded `rate`, `positive`, `count` or `duration` on
`[0, ∞)`), asking for bounds or a different rule; `from_prior` draws such a
parameter from its prior. A rule applies only to parameters in `[estimate]`;
anything in `[fixed]` takes its declared value.

**The default.** With no `starts` in the file and no `--starts` on the command
line, `fit run` resolves the rule against the problem before the identity is
taken (`Method::resolve_starts`): `from_prior` when every estimated parameter's
resolved prior is a distribution the chains can be drawn from,
`uniform_unconstrained` otherwise. The startup block says which and why —
`starts:   from_prior (one independent draw per chain) — default: every
estimated parameter declares a prior`,
or `… uniform_unconstrained … —
default: no sampleable prior on beta`. A file
that spells the default out hashes the same as one that omits it, because the
hashed payload always carries the resolved rule.

**Handles.** A sourced rule names its source run by `@<label>` (§6.8), a prefix
of the run's `run_id`, or the run's folder. A `fit.toml` path names no run and
is refused with the runs it made listed (§2.2.1). The source is resolved before
the model is loaded, and its stored convergence verdict is checked where it is
consumed: a `from_mle` or `from_posterior` source whose verdict is not converged
is refused, naming the verdict, unless `--allow-nonconverged-source` is passed
(§7.3).

`starts` is identity-bearing: the resolved rule is in the run's identity, and a
sourced rule's consumed file (the source run's `fit_state.toml` or `draws.tsv`,
a params TOML) is a dependency of the run on that file, so two runs differing
only in `starts` are two results and a regenerated source re-keys its consumer.
A sourced rule enters the identity by its kind (`from_mle`, `from_posterior`,
`from_params`) and its source, not by its handle as typed: `@scout`, the scout's
folder and a prefix of its `run_id` name one source and key one run. A
`--starts` file named directly is stored in the run as `inputs/starts.<ext>` and
keyed by its bytes. The handle as typed is kept, unhashed, in the run's
`run_inputs.json` (`starts_source`).

**A refused spread start is redrawn** (gh#887). A spread rule is a lottery over
one draw, so a start the filter cannot score is one unlucky draw rather than a
verdict on the chain: it is redrawn under the same rule, at a seed derived from
the run's seed and the attempt index, up to `MAX_START_ATTEMPTS = 10` per chain,
and the chain is refused only when every attempt fails. A point rule has nothing
to redraw and gets one try; so do `uniform`, `lhs` and `uniform_unconstrained`
at one chain, whose single draw is the base point. IF2 scores every chain's
start with one particle-filter pass before the swarm is perturbed
(`runner::preflight_spread_starts`); PGAS retries on a start whose complete-data
log-posterior is still non-finite after the first trajectory update; PMMH
retries on the init-eval's `PFDegenerate`. Each attempt is announced
(`chain 0: start 3 of 10 refused (…); drawing another`) and the refusal, when it
comes, says how many were tried: `none of 10 starts drawn
under`starts =
from_posterior
<file>`could be scored by the filter; the
last: EssCollapsed at obs_window=…`.
Skip-and-continue after a refusal is unchanged (§10.7).

**What a run records.** Every multi-chain sampler and IF2 write
`chain_starts.tsv` — the point every chain actually began at, before any
perturbation, and every attempt a retry rejected on the way there — under a
header naming the rule and counting the rejected draws
(`# camdl chain_starts;
starts=from_mle @scout; chains=4; kind=point; retried=0`).
Its columns are `chain_id` (the number of the `chain_N/` directory beside it and
the number every stderr refusal and `bad_init` record uses), `attempt`
(0-based), `status` (`accepted` — the start the chain ran from; `rejected` — a
draw the filter could not score, a redraw followed; `refused` — the last
attempt, also unscoreable, the chain did not run; `skipped` — the chain ran from
this start and was dropped before the end, so the fit's results do not include
it), `source`, one column per estimated parameter, then `ess` (the filter's ESS
at refusal, blank otherwise) and `reason` (blank for an accepted row; for a
skipped row, why the chain was dropped); one row per attempt in (chain, attempt)
order, so a healthy run has one `accepted` row per chain at attempt 0. The file
is written before the sampler runs and, when IF2 skips a chain mid-run (its
particle-filter degeneracy watchdog fired), written again with that chain's row
marked `skipped`. That file, not the config, is the authority on what a run did.
`fit_state.toml` records the count as `n_chains` (started) and `n_good_chains`
(reached the end); an IF2 run whose `n_good_chains` is below `n_chains` has the
gate verdict `incomplete` (§13.1), not `not_assessed`, even when one chain is
left. `fit_state.toml` carries the same rule as `chain_init_source` beside
`chain_starts_kind = "spread" | "point"`, and `fit summary` prints both under
the leaf's identity line, with the run id the handle resolved to when the run
started (`run_inputs.json`'s `starts_source.run_id`):

```
seeded from:  from_mle @scout, run 4b1e0c9a2f77 (every chain at one point)
```

A multi-chain sampler leaf whose kind is `point` is read with its R̂ withheld:
`fit summary` reports
`R̂ — not assessed: all 2 chains started at one point
(starts = from_mle @scout)`
in place of a verdict, `camdl predict` stamps its bands the same way, and the
ESS — a within-chain quantity — is kept. Writing the point rule is the
acknowledgement; `fit run` says so once at startup for a sampler with
`chains > 1`. An IF2 run keeps its Â band under the same header, since IF2's
perturbation re-spreads the swarm after the start.

**Base-point precedence** (`runner.rs`), lowest to highest: the model's declared
value, `[fixed]`, `[estimate].start`. A sourced point rule replaces the base
point for every estimated parameter it carries.

### 6.7 `[provenance]` — lineage metadata

```toml
[provenance]
derived_from = "fits/01_all_free.toml"
reason = "free the reporting fraction"
```

Both fields are optional strings (`FitProvenance`, `config_v2.rs`). No runner
reads them; `camdl fit new` writes `derived_from` when deriving one config from
another, and the block is otherwise for human navigation.

It is not part of any identity: the problem's identity holds the typed fields of
§2.2, so editing `reason` reuses the stored fit. Each run keeps the file it was
given as its `fit.toml`, so the text is kept with the run.

### 6.8 Where output goes

Fits are content-addressed in the run store. There are no configured output
paths beyond the root; the folder names are labels and hashes of the inputs, in
the layout of §2.2:

```
<root>/<build>/fit/<model stem>-<12 hex>/          the fit problem
        problem.toml  partition.json  inputs/  held_out/  model.camdl
    runs/<method>-<12 hex>/                        the method's settings
        seed_<n>-<12 hex>/                         one fit run
            fit.toml  method.toml  run_inputs.json  fit_state.toml  chain_starts.tsv
            chain_0/ … chain_{N-1}/
```

**One index base.** Every index camdl shows a user counts from 0 — this is the
single rule, stated once here and relied on everywhere below. A fit's chains are
`0 … N-1` in every place that names one: the `chain_N/` directories, the
progress and refusal lines, `chain_starts.tsv`'s `chain_id`, `fit_state.toml`'s
`best_chain` and `chain_eval_ids`, `run_inputs.json`'s `best_chain`,
`mle_params.toml`'s `[provenance] chain`, every `fit summary` table,
`--exclude-chains`, the `chain` column of `draws.tsv` and `trajectories.tsv`,
the `chain` column `camdl predict --by-chain` adds, `progress.json`'s per-chain
rows and `cross_chain_compat.json`'s `chains`. `simulate`'s `replicate`, `draw`
and `point_id` count from 0 too. Joining any two of these takes no arithmetic,
and no artifact carries a second base to convert from.

- **`<root>`** is the run store's root (§2.1).
- **The problem** is labelled by the model file's stem; its 12 hex digits are of
  the problem's identity (§2.2). Two files that pose one problem and differ in
  `[method]` share it.
- **`<method>`** is the algorithm's name; the group's 12 hex digits are of the
  method's settings without its length.
- **The run** is per seed, labelled by the seed typed. Its `run.json` records
  its status and files; its label is not in any record of the run (below).

Per-algorithm leaf contents: IF2 writes `mle_params.toml`, `final_params.toml`,
`diagnostics.tsv`, `chain_evaluations.tsv`, and
`chain_N/{final_params.toml, parameter_traces.tsv}`. The samplers write
`draws.tsv`, `<algorithm>_summary.json` (`pgas_summary.json`,
`pmmh_summary.json`, `mh_summary.json`), and `chain_N/trace.tsv`; PGAS and PMMH
also write `chain_N/resume_state.bin`. The NLopt optimizers write
`mle_params.toml` and `chain_results.tsv`, and beside them the live search trace
`opt_trace.tsv`, a record of the attempt outside the manifest (§10.13). A
`pfilter` method with `record_prequential` writes `prequential.{tsv,json}`.

Every `<algorithm>_summary.json` carries `schema = "camdl.fit-summary/v2"`, and
every reader requires it: a file without the tag, or with a tag this camdl does
not know, is refused by name rather than read. The tag exists because two of its
keys changed _meaning_ under unchanged names — `rhat` was the classic
Gelman–Rubin statistic and is now
`max(rank-normalized split-R̂, folded split-R̂)`; `ess` was a sum of per-chain
Geyer estimates suppressed to NaN above R̂ 1.1 and is now the cross-chain
bulk-ESS, never suppressed. Absence of the tag therefore means the pre-tag
vintage, and nothing else in the file distinguishes the two. Re-running a fit
refreshes it; a stored fit from before the tag is read by the camdl that wrote
it.

`v2` is the same statistics under one renamed key: the algorithm is named by
`method`, not `stage`. A rename under an unchanged tag is the vintage problem
the tag exists to prevent, so it is a version rather than a silent edit.

A completed run is reused on a second identical invocation
(`cache hit — reusing <folder>`); pass `--force` to re-run it: the stored run is
moved to quarantine and the result stored afresh.

Fits are addressed downstream by handle: `@label`, a prefix of a run's or a
problem's id, or a stored folder. A run handle names one run; a problem handle
(its id or folder) names the problem. A folder handle is read from the store
that holds it, the nearest directory above it with a `store.json`, whatever
`CAMDL_STORE` names. A handle to a run that did not complete (still running,
failed, interrupted or damaged) is refused with the run's folder, its state and
the reason the store recorded:

```
error: camdl-store/0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-797c0b53281a/runs/pmmh-85f985a8c2a2/seed_1-37ee86900a72 names a fit run that did not complete and is not read: camdl-store/0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-797c0b53281a/runs/pmmh-85f985a8c2a2/seed_1-37ee86900a72 was interrupted: its process ended before it completed
```

A command that needs a run (`fit summary`, `camdl predict`, `compare`,
`simulate --draws posterior --fit`, a sourced `starts`, `--resume`) refuses a
problem handle, listing the problem's runs
(`error: 8bcb names a fit problem, and this reads one run of it. Its runs:`),
and refuses a `fit.toml` path with the runs it made listed (§2.2.1). A command
that needs only the problem (`pfilter --fit`, `design`, `profile --fit`) accepts
either kind of handle or a `fit.toml`. `camdl show` and `camdl cat` resolve
through the run store's handles (§4.5): `show` on a fit run prints its record,
and `cat <run> <name>` prints one of its files.

**Labels.** `fit run --label L` names the invocation, as every command on the
run store does (`docs/dev/proposals/2026-11-20-run-store-api.md`, "Labels"); no
record of the run holds the label. `@L` is the latest invocation labelled `L`
whose results all completed, and `@L~n` counts back through such invocations. A
rerun of the same command with `--label L` reuses the stored run and makes it
`@L`:

```
$ camdl fit run fit_if2.toml --seed 3 --label i
…
── method: if2 ──
  cache hit — reusing 0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-8bcb39805e9d/runs/if2-e5e94eb7c269/seed_3-be6657446294
```

A run started from a labelled source (`from_mle`, `from_posterior`) records the
handle as typed and depends on the source run's file, so moving the label cannot
change what an existing run was started from; rerunning it after the move
resolves the new source, a new dependency, and re-keys.

### 6.9 Load-time validation

`FitConfig::from_toml_str` (`config_v2.rs`) runs the checks below that read the
raw file before the typed parse and the `[method]` key check on its result;
`FitConfig::validate` then runs the semantic checks once the model IR is loaded
and `[fixed]
from_scenario` has been expanded. Everything below fires before the
method executes.

**Parse-time.**

- A `[stages]` table. There is no silent conversion: the loader prints the
  rewrite, in the file's own declaration order, naming the last-declared stage
  as the file's `[method]` and every other stage as a file of its own. A chained
  stage (`init_mle`) is a decision the message hands back, because the file
  cannot name a handle that does not exist until the upstream has run:

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
    `init = "uniform_unconstrained"` becomes `starts = "uniform_unconstrained"`
    See `camdl docs fit-toml`.
  ```

  An `init = "survey_top_k"` (or a `survey_path` / `survey_top_k_n`) adds
  `` `init = "survey_top_k"` … was removed with the command that wrote its
  input. Use `starts = "from_prior"`, or run a short fit and write `starts
  = { from_posterior = "@handle" }`. ``
  An empty `[stages]` is refused with `replacement: delete it`.

- An unknown top-level key:

  > `` unknown field `bogus`, expected one of `model`, `data`, `synthetic`, `fit_seeds`, `simplex_groups`, `store`, `estimate`, `fixed`, `method`, `config`, `simulate`, `scenario`, `enable`, `disable`, `ic_free`, `provenance` ``

- An unknown key inside `[method]`. `Algorithm` is internally tagged, so serde
  cannot deny unknown fields on it; a post-parse pass compares the raw keys
  against the set the parsed method serializes back to, plus `starts`
  (`validate_method_keys`, `config_v2.rs`):

  > ``unknown key `chains` in [method] (algorithm = "pfilter").``
  > `allowed keys: algorithm, backend, particles, record_ancestry, record_prequential, replicates, starts`

  The keys the split retired name their replacement instead of "unknown": `init`
  → `` `init` is now `starts`: write `starts = "lhs"`. ``, `init_mle` →
  `` `init_mle` has no replacement key: a warm start from a stored fit is
  `starts = { from_posterior = "@handle" }` (one draw per chain, keeps R̂
  meaningful) or `starts = { from_mle = "@handle" }` (every chain at one point,
  R̂ not assessed). ``,
  and `survey_path` / `survey_top_k_n` → the survey message above. The two NLopt
  variants are newtype-wrapped structs and get plain serde `deny_unknown_fields`
  for their own keys:
  `` unknown field `zzz`, expected one of `backend`, `chains`, `tolerance`, `max_evals`, `gate` ``.

- Unknown enum values — an unrecognized `algorithm` (via a missing tag),
  `backend`
  (`` unknown variant `gillespie`, expected `chain_binomial` or
  `ode` ``), or
  a prior whose field names do not match a catalogue entry
  (`data did not match any variant of untagged enum EstimatePriorSpec`); an
  unknown `starts` rule
  (``unknown starts rule `foo`; expected one of uniform_unconstrained, lhs, uniform, from_prior, single, or a sourced rule { from_posterior = "@handle" }, { from_mle = "@handle" }, { from_params = "params.toml" }``).

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
   > ``parameter `beta` is in both [estimate] and [fixed]; a parameter is either estimated or given a value, not both.``
   > `parameters neither estimated nor fixed: I0`
   > `parameters not in model: nope`
8. **(algorithm, backend)** against the registry. The error names the structural
   reason, suggests the right alternative, and lists the supported pairs:
   > `[method]: method has algorithm = "if2" with backend = "ode", which is not a supported inference method.`
9. `ic_free` support for the method.
10. IF2 `iterations ≥ 1`.
11. `burn_in < iterations` (`pmmh`, `mh`) and `burn_in < sweeps` (`pgas`), using
    the **defaults** when unset — so a short sampler run must set `burn_in`
    explicitly:
    > `[method]: burn_in (2000) ≥ sweeps (5) — every sample is discarded as burn-in, so the fit retains no posterior draws (and the post-burn acceptance rate degenerates to 0%). Reduce burn_in or raise sweeps. (burn_in defaults to 2000 when unset.)`
12. Non-empty `bounds` on every entry that declares them.
    > `estimate.beta: bounds [0.6, 0.15] are empty (lo must be < hi)`
13. Simplex groups: ≥ 2 members, every member in `[estimate]`, no member in two
    groups, no negative lower bound. A member's `[method.perturbation]` entry
    may not set `initial_state`, which is refused when the steps are resolved:
    the group perturbs its members at t = 0 only.

`validate_priors_present` runs next, with the model IR in scope so the `~`
fallback is honored — for a sampler method only; priors an optimizer ignores are
the normal case under one method per file and draw no warning. Then one note
that does not stop the run: a posterior sampler under a point rule (`single`,
`from_mle`, `from_params`) with `chains > 1`, whose R̂ `fit summary` will report
as not assessed (§6.6).

Two checks fire later, at method build rather than at load: the
bounds-may-narrow-not-loosen rule (`runner.rs`) and `burnin_dt` validity
(`config_v2.rs`).

A file with no `[method]` passes every check above for the readers that take
`--fit`; `fit run` alone refuses it:

> `this fit.toml declares no`[method]`table, so there is nothing to run. It is a complete problem for`simulate
> --fit`,`pfilter --fit`and`profile --fit`; to fit it, add …`

## 7. The Fit CLI

`camdl fit` is a group of six subcommands. One runs inference; two read what a
run left behind; two work on configs; one is a static capability listing. A
prediction from a stored run is the top-level `camdl predict RUN` (§2.2.2).

| subcommand    | what it does                                                                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fit run`     | run the `[method]` of a `fit.toml`                                                                                                                     |
| `fit summary` | render one fit's convergence verdict, θ̂ table and provenance checks; `--explain` defines each column, `--parameters` adds the model's parameter legend |
| `fit table`   | walk a run store and render one row per fit run                                                                                                        |
| `fit diff`    | compare two `fit.toml` configs                                                                                                                         |
| `fit new`     | scaffold a new `fit.toml` derived from an existing one                                                                                                 |
| `fit methods` | list the supported (algorithm, backend) pairs                                                                                                          |

There is no `camdl fit list` and no `camdl fit where`. Browsing every cached run
— fits, simulations, profiles — is the top-level `camdl list`. A fit run is
named when it runs, with `fit run --label` (§6.8).

### 7.1 Invocations

The commands below use the worked project of §8 (`models/`, `data/`, `fits/`,
with fits stored in `camdl-store/`).

```bash
# Run the file's [method].
camdl fit run fits/01_mle.toml --seed 1

# Long fits: force plain progress lines and capture them.
camdl fit run fits/01_mle.toml --seed 1 --progress plain 2>&1 | tee fit.log

# Sweep a [fixed] parameter; repeat the flag for a Cartesian grid.
camdl fit run fits/01_mle.toml --sweep "rho=0.5,0.6"
camdl fit run fits/01_mle.toml --sweep "rho=0.5,0.6" --sweep "k=5,10"
camdl fit run fits/01_mle.toml --sweep "k=lin(5,15,3)"
camdl fit run fits/01_mle.toml --sweep "k=log10(1,100,3)"

# Tag the run so it is findable later, and by another file's `starts`.
camdl fit run fits/01_mle.toml --label mle

# Start the chains from another fit, overriding the file's `starts`.
camdl fit run fits/09_pgas_only.toml --starts from_mle=@mle
camdl fit run fits/09_pgas_only.toml --starts from_posterior=@short-run
camdl fit run fits/09_pgas_only.toml --starts from_mle=089bc0f3   # a run id prefix

# Read side.
camdl fit summary @mle
camdl fit summary '@default steps, take 1'
camdl fit summary @mle --params-only
camdl fit summary @mle --explain      # define each column
camdl fit summary @mle --parameters   # the parameter legend
camdl fit table camdl-store
camdl fit table camdl-store --with-method pgas --format md
camdl fit diff fits/01_mle.toml fits/02_posterior.toml
camdl fit new --from fits/01_mle.toml fits/04_derived.toml
camdl predict @posterior --n-draws 20
camdl fit methods
camdl list
```

**Fit handles.** `fit summary` and `camdl predict` take a _handle_ (the
positional argument RUN) naming one run: `@<label>`, a run id prefix, or the
run's folder (§6.8). A `fit.toml` is refused, with the runs it made listed:

```
$ camdl fit summary fit_pmmh_all.toml
error: fit_pmmh_all.toml is a fit configuration, and a configuration never names a run: one file makes several runs whenever its data, its seeds, a sweep or the build change.
    9cd433e2dedb  camdl-store/0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-797c0b53281a/runs/pmmh-85f985a8c2a2/seed_1-9cd433e2dedb
  Name one by its id or folder, or by `@label` (`fit run --label`).
```

**Chaining files by handle.** A `fit.toml` carries one `[method]`, so a pipeline
— an IF2 scout that finds the basin, a PGAS file that samples it — is two files,
and the second names the first through its `starts`:

```toml
# fits/02_posterior.toml
[method]
algorithm = "pgas"
…
starts = { from_mle = "@scout" }
```

The handle is resolved when the second file runs, so the scout has to have run
(and been labelled) first; a handle that names nothing, or a source that did not
converge, is refused where it is consumed (§7.3). A completed run is served from
the store, so re-running the first file is a cache hit:

```
$ camdl fit run fit_if2.toml --seed 3 --label i
…
── method: if2 ──
  cache hit — reusing 0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-8bcb39805e9d/runs/if2-e5e94eb7c269/seed_3-be6657446294
```

A `fit.toml` with a `[stages]` table is refused (§6.9).

### 7.2 CLI Type

`fit run` takes the config path plus four groups of flags: run control, chain
starts, the dt-convergence audit, and per-algorithm overrides. The run-control
core:

```rust
pub struct FitRunArgs {
    /// Fit configuration file (v2 TOML)
    pub config: PathBuf,

    /// Where the chains begin, overriding the file's `starts`: a rule name,
    /// or `name=source` for a sourced rule (`from_mle=@scout`).
    #[arg(long, value_name = "SPEC")]
    pub starts: Option<ChainStarts>,

    /// Start from a `from_mle` / `from_posterior` source whose stored
    /// verdict is not converged.
    #[arg(long)]
    pub allow_nonconverged_source: bool,

    /// Rayon thread cap; 0 = all logical cores. Bit-identical regardless.
    #[arg(long, default_value_t = 0, env = "CAMDL_PARALLEL")]
    pub parallel: usize,

    /// RNG seed (default: 1)
    #[arg(long)]
    pub seed: Option<u64>,

    #[arg(long)]
    pub force: bool,

    /// Extend a completed PGAS/PMMH run from a base run, addressed by
    /// `@label`, run_id prefix or run folder.
    #[arg(long, value_name = "BASE_REF", conflicts_with = "force")]
    pub resume: Option<String>,

    /// Cartesian sweep over a [fixed] parameter (repeatable).
    /// SPEC is `V1,V2,...` | `lin(min,max,n)` | `log10(min,max,n)`.
    #[arg(long, value_name = "NAME=SPEC")]
    pub sweep: Vec<SweepSpec>,

    /// The integration step, written into the problem's `[config] dt`
    /// under the rule of §1.3.1.
    #[arg(long)]
    pub dt: Option<f64>,

    /// Display label, 1–64 chars of `[A-Za-z0-9 ,._-]`.
    #[arg(long, value_name = "TEXT")]
    pub label: Option<String>,

    /// The run store's root, over the fit file's `store` key (§2.1).
    #[arg(long, value_name = "DIR")]
    pub store: Option<PathBuf>,
}
```

Note the shapes. `--resume` carries a **base reference**, not a boolean;
`--parallel` is a plain `usize` with a `0` default meaning "all cores"; and
there is **no `--skip-chains`** (chain exclusion is a read-side view:
`fit summary --exclude-chains`, `camdl predict --exclude-chains`,
`fit table --exclude-chains`).

`--force` on `fit run` recomputes a run that already exists: the stored run, its
predictions included, is moved to `<root>/.quarantine/` (never deleted) and the
fresh result is stored at the same address. Without the flag the same command
reports a cache hit and exits 0. A run another live process is still writing is
not displaced; the run stops with an in-progress error naming the owner.
`simulate` takes `--force` with the same meaning. `pfilter --force` computes the
run and its by-products again. The completed result is moved to
`<root>/.quarantine/` before the recompute starts, so a recompute that fails
leaves nothing at the result's address; the old result is recovered from
`.quarantine/`. A run held by an owner not known to be stopped is refused
without the flag, naming the owner's host and `--force`. With it, the run is
taken unless its owner is on this host and running, or on another host with a
beat that changes; an owner on another host whose beat is stale or cannot be
read is taken, so pass `--force` there only once that process is known to have
stopped. `profile` has no `--force`.

`--resume <BASE_REF>` extends a completed PGAS or PMMH run: the base run is read
read-only and the longer chain is written to a new run beside it, under the same
method group, keyed on the new `sweeps` / `iterations` with a dependency on the
base — a distinct deterministic result, not bit-identical to an uninterrupted
fit of the same length. The new run holds the whole chain, the base's rows and
the new ones, and records where each segment begins (§9.5, "A resumed run").
Running the same resume again, of the same base to the same length, is a cache
hit on that run, as any identical invocation is. The file's `starts` (or
`--starts`) is resolved as on any run, so a file that starts from another fit
resumes like any other:

```
$ camdl fit run fit_pmmh_all.toml --label post
   stored pmmh · 0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-797c0b53281a/runs/pmmh-85f985a8c2a2/seed_1-9cd433e2dedb · 0.3s

# raise `iterations` from 20 to 40 in a copy of the file, then extend that run
$ camdl fit run fit_pmmh_40.toml --resume @post
  chain 0: resuming from step 20
  chain 1: resuming from step 20
   stored pmmh · 0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-797c0b53281a/runs/pmmh-85f985a8c2a2/seed_1-a372ef8992fa · 0.3s
```

**Chain starts.** `--starts <SPEC>` overrides the file's `starts` with the same
rules §6.6 lists — a bare name, or `name=source` for a sourced rule:

| `--starts`                | where the chains start                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------ |
| `from_prior`              | one draw per chain from each parameter's resolved prior (the default when every one has a prior) |
| `uniform_unconstrained`   | i.i.d. U(−2, 2) on the unconstrained scale, mapped inward (the default otherwise)                |
| `lhs`                     | Latin-hypercube stratified inside bounds                                                         |
| `uniform`                 | per-chain uniform draw inside `[estimate]` bounds                                                |
| `from_posterior=<handle>` | one row per chain from a fit's `draws.tsv` (or a draws TSV path)                                 |
| `single`                  | every chain at the seeded base point                                                             |
| `from_mle=<handle>`       | every chain at a fit's estimate                                                                  |
| `from_params=<toml>`      | every chain at the point in a flat params TOML                                                   |

The override is keyed into the run's identity like the file's own `starts`, and
a sourced rule's source is refused when its stored verdict is not converged
unless `--allow-nonconverged-source` is passed.

**The dt audit.** `--no-dt-check` skips the post-fit Richardson dt-convergence
check, `--dt-check-strict` tightens its threshold (0.5 nats for chain_binomial,
0.1 for ode_rk4, against routine defaults of 2.0 / 0.5), and
`--dt-check-halvings <N>` sets how many halvings it evaluates (default 2).

**Per-algorithm overrides.** Each writes through to the method identity, so an
overridden run is a distinct artifact rather than a cache hit on the
un-overridden one. A flag is accepted only by the methods in its row; on any
other method `fit run` refuses it before any work, naming the flag, the method,
and the methods that read it.

| flag                      | overrides                           | methods that read it                   |
| ------------------------- | ----------------------------------- | -------------------------------------- |
| `--decibans-thresh <DB>`  | `[method.gate].decibans_thresh`     | if2, nl-sbplx, nl-bobyqa, nl-lbfgs     |
| `--no-dt-check`           | `[method.dt_check].enabled = false` | if2, mh, nl-sbplx, nl-bobyqa, nl-lbfgs |
| `--dt-check-halvings <N>` | `[method.dt_check].n_halvings`      | if2, mh, nl-sbplx, nl-bobyqa, nl-lbfgs |
| `--dt-check-strict`       | `[method.dt_check].threshold_nats`  | if2, mh, nl-sbplx, nl-bobyqa, nl-lbfgs |
| `--cooling-target-iters`  | `cooling_target_iters`              | if2                                    |
| `--max-tree-depth <N>`    | `max_tree_depth`                    | pgas                                   |
| `--trajectory-warmup <N>` | `trajectory_warmup`                 | pgas                                   |
| `--csmc-sweeps-per-nuts`  | `csmc_sweeps_per_nuts`              | pgas                                   |
| `--n-trajectories <N>`    | `n_trajectories`                    | pgas                                   |
| `--diagonal-mass`         | `dense_mass = false`                | pgas                                   |
| `--no-nuts`               | `use_nuts = false`                  | pgas                                   |
| `--no-ancestor-sampling`  | `ancestor_sampling = false`         | pgas                                   |
| `--binomial <SAMPLER>`    | `binomial`                          | pgas, pmmh, if2, pfilter               |
| `--no-adapt`              | `adapt = false`                     | pmmh, mh                               |
| `--adapt-start <N>`       | `adapt_start`                       | pmmh, mh                               |
| `--record-ancestry`       | `record_ancestry = true`            | pfilter                                |
| `--record-prequential`    | `record_prequential = true`         | pfilter                                |

The boolean overrides are one-way: they can switch a setting off, and switching
it back means editing the TOML.

A flag the method reads is also refused when the method's other settings leave
it nothing to do: `--max-tree-depth` or `--diagonal-mass` with NUTS off
(`--no-nuts` or `use_nuts = false`), `--adapt-start` with adaptation off
(`--no-adapt` or `adapt = false`), `--dt-check-halvings` or `--dt-check-strict`
with the dt-check off, and `--dt-check-strict` beside a declared
`[method.dt_check] threshold_nats`.

**Seeds and fits.** A fit runs at a single base seed — `--seed N`, default 1 —
and derives each chain's seed from it by a golden-ratio mix,
`base ^ (chain_index · 0x9e3779b97f4a7c15)`, not by addition. For a
start-sensitivity sweep, the `fit.toml` also accepts a top-level `fit_seeds`
list; each entry becomes its own run:

```toml
fit_seeds = [1, 2]
```

```
━━━ fit_seed=1 ━━━
   stored if2 · 0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-8bcb39805e9d/runs/if2-e5e94eb7c269/seed_1-556beb126546 · 0.8s
━━━ fit_seed=2 ━━━
   stored if2 · 0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-8bcb39805e9d/runs/if2-e5e94eb7c269/seed_2-603b688f7602 · 1.1s
```

The seed is in the run's identity and not the problem's, so adding seeds adds
runs to a problem rather than re-keying it.

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

Repeating the flag takes the Cartesian product, and each point runs the method
independently on its own problem (§2.3):

```
$ camdl fit run fits/01_mle.toml --sweep "rho=0.5,0.6" --sweep "k=5,10"
sweep: 4 points
═══ sweep point 1/4: rho_0.500__k_5.000 ═══
          best ll=-59.9 (chain 1) in 8.6s
═══ sweep point 2/4: rho_0.500__k_10.000 ═══
          best ll=-59.3 (chain 3) in 8.7s
═══ sweep point 3/4: rho_0.600__k_5.000 ═══
          best ll=-59.6 (chain 3) in 8.8s
═══ sweep point 4/4: rho_0.600__k_10.000 ═══
```

**Source convergence.** A file that starts from a stored fit
(`starts = { from_mle = "@scout" }`, `{ from_posterior = "@short" }`, or the
`--starts` override) is checked against that source's stored verdict where the
handle is resolved, before the model is loaded. An IF2 source is judged by its
own gate — the tail chain-agreement Â leg and the inter-chain log-likelihood
spread in decibans — an `nl-*` source by the verdict its optimizer recorded at
the end of its run (every chain converged, the chains' estimates agree, and
their log-likelihoods agree in decibans), and a sampler source by the same R̂
classification `fit summary` renders. A source that did not converge is refused
with the verdict, so a warm start cannot launder one fit's multi-modality into
the next:

```
$ camdl fit run fits/02_posterior.toml
error: starts source @scout: it did not converge.

  Tail Â (IF2 chain agreement over the last half of iterations), threshold 1.01:
    ✗ gamma      Â =  1.145   (>= 1.01)
      beta       Â =  0.965

  Loglik spread: 0.2 (best chain loglik -58.4)

  Failing: gamma (Â=1.15)

  Pick one:
    - re-run it with more chains or iterations
    - narrow bounds to the basin its best chain found
    - fix a weakly identified initial-state parameter rather than estimate it

  To start from it anyway (the starts may launder multi-modality), re-run
  with --allow-nonconverged-source
```

Every estimated parameter's Â is judged, initial-state parameters (IF2's
`initial_state = true`) included, and an Â that is not finite (within-chain
variance collapsed) makes the verdict incomplete, which is refused like a
failure: the decibans leg does not stand in for it, since chains spread along a
ridge can agree in log-likelihood while disagreeing in θ. An IF2 source that
lost started chains (`n_good_chains < n_chains`) is incomplete too, and refused
with a message naming the count (`5 of its 6 IF2 chains were lost …`), what that
leaves to judge, the remedy (re-run the source so every chain finishes) and the
`--allow-nonconverged-source` override. When such a source also fails a leg, the
failure is the verdict and the refusal adds the lost-chain count. A source that
asked for one chain is accepted, as before.

`--allow-nonconverged-source` downgrades the refusal to a warning that repeats
the verdict and says `starting from it anyway`. A source between the soft and
hard agreement bands is accepted with a warning naming the parameters its chains
only nearly agreed on. There is no gate after the run: a regression is visible
in `fit table`, and the consumer's own verdict is its own.

Under `--sweep` every point resolves the same source, so a refused source
refuses the whole sweep before any point runs.

For a dedicated profile-likelihood workflow — fix a focal parameter, maximise
over the rest at each grid point — use `camdl profile`, which runs IF2 per cell
over its own `--sweep` grid.

---

## 8. fit.toml Examples

The configs in §8.1 to §8.3 were executed end to end, and their output is shown,
with each stored folder written `<build>/fit/sir-…/runs/<method>-…/seed_1-…`;
§8.4's config is shown without output. They share one project layout; every path
inside a `fit.toml` resolves **relative to that file**, which is why the configs
in `fits/` reach back out with `../`.

```
models/sir.camdl          # closed SIR, weekly NegBinomial reported cases
data/cases.tsv            # time, weekly_cases
params/sir_fixed.toml     # shared [fixed] block
fits/*.toml
camdl-store/              # the run store, the fit files' `store` key
```

The iteration counts are deliberately small so each example finishes in seconds.
Real fits want one to two orders of magnitude more of everything.

### 8.1 MLE with IF2

The minimum useful fit: an IF2 method maximising the stochastic-process
likelihood.

```toml
store = "../camdl-store"

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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 200
iterations = 10
cooling = 0.7
```

```
$ camdl fit run fits/01_mle.toml --seed 1 --label mle
   cached IR for sir.camdl (97c3d4c1)
fit: fits/01_mle.toml (method=if2)
  model:    fits/../models/sir.camdl
  estimate: beta, gamma
  fixed:    N0, I0, rho, k
  starts:   uniform_unconstrained (one independent draw per chain) — default: no sampleable prior on beta, gamma
  output:   fits/../camdl-store

── method: if2 ──
  note: `starts = "uniform_unconstrained"` draws every chain's start, so `[estimate].start` is unused here for: beta, gamma. Use `starts = "single"` to start every chain at the declared values, or drop the `start` entries.
running 4 chains × 200 particles × 10 iterations, cooling=0.7, dt=1

transforms (chain 1 of 4; chains start at different points — see chain_starts.tsv):
  beta         log_interval  [0.05, 1]  start=0.3757 z=0.72  step(z)=0.050 ≈ 0.0093 natural at start 0.25 (default)
  gamma        log_interval  [0.01, 0.5]  start=0.1405 z=0.73  step(z)=0.050 ≈ 0.0047 natural at start 0.1 (default)

  ⚠ 2/2 parameters use the default step. Check traces, and write a step in the method's table where it does not move.

cooling: cf50=0.70, reached at iter 50 (target), over a 10-iteration run × 12 observed times
  iter   1: rw_sd at 99.3%
  iter  50: rw_sd at 70.0% (cf50 reached)
  iter  10: rw_sd at 93.1% (run end)

evaluating loglik (every 10 iterations, all 4 chains)...
  chain 0: ll=-58.4      chain 1: ll=-58.4      chain 2: ll=-58.6      chain 3: ll=-58.5

loglik-eval: re-scoring final-iter θ̂ (4 chains × 8 replicates @ 4000 particles)...

best chain: 1 (loglik=-58.38 ± 0.01)
chain clean logliks: [-58.5, -58.4, -58.6, -58.6]

Â:
  beta         Â=0.965 ✓
  gamma        Â=1.145 ~

dt-convergence at θ̂: PASS  (|Δ_leg1| = 0.65, |Δ_leg2| = 1.16 nats (vs τ = 2.00); converged.)

── diagnostics ──────────────────────────────────────
  i Auto step for 'beta': 0.0500 in its transformed coordinate.
  i Auto step for 'gamma': 0.0500 in its transformed coordinate.
  ! 1/2 parameters have Â > 1.1 (max 1.15).
  0 error(s), 1 warning(s), 2 info

   stored if2 · <build>/fit/sir-…/runs/if2-…/seed_1-… · 8.5s
          best ll=-58.4 (chain 1) in 8.5s
```

Two things the run says that are worth reading. The `starts` line and its
`note`: no parameter here declares a prior, so the default rule is
`uniform_unconstrained`, which draws each chain's start, and the `start` values
in `[estimate]` are inert unless you set `starts = "single"`. And the `Â` block:
chain agreement on `gamma` is 1.145, which at ten iterations means the chains
have not agreed on a basin. `camdl fit summary` renders the same run as a
verdict:

```
$ camdl fit summary @mle

camdl-store/<build>/fit/sir-…/runs/if2-…/seed_1-…/
  camdl 0.1.0+3e2b2888 · if2 · 4 chains
  seeded from:  uniform_unconstrained (one independent draw per chain)

  best loglik:  -58.4 (if2)  (loglik-eval, max across chains)

  compound convergence gate
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

The gate verdict here is informational — nothing consumes this fit. It becomes
binding the moment another file starts from it
(`starts = { from_mle = "@mle"
}`, §8.2): that file is refused until the scout
converges, or `--allow-nonconverged-source` is passed (§7.3).

### 8.2 MLE, posterior sampling, and a filter evaluation

A pipeline of three files: IF2 finds the basin, PGAS samples the posterior
starting from it, and a particle filter re-scores θ̂ with a replicate spread.
Each file carries one `[method]`; the second and third name the first by the
label it was run with.

Priors are **required** for a sampler. They are externally-tagged inline tables
keyed by the distribution name —
`prior = { log_normal = { mu = …, sigma
= … } }`, not
`prior = { dist = "log_normal", … }`. A `fit.toml` prior overrides the model's
`~ <dist>(...)` declaration; if neither is present, camdl warns and falls back
to flat, and `prior = { flat = {} }` is how you ask for flat on purpose without
the warning.

The problem half is the same in all three files; the scout is `01_mle.toml` with
priors added:

```toml
# fits/02_scout.toml
store = "../camdl-store"

[provenance]
derived_from = "01_mle.toml"
reason = "priors, so PGAS can sample the basin scout finds"

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

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 200
iterations = 10
cooling = 0.7
```

The posterior and evaluation files repeat the problem half and change only
`[method]`; `camdl fit new --from fits/02_scout.toml fits/02_posterior.toml`
copies it. Because the problem is the same, all three runs land in one problem
folder, in sibling method groups.

```toml
# fits/02_posterior.toml — [method] only; the rest as 02_scout.toml
[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 50
sweeps = 200
burn_in = 50
starts = { from_mle = "@scout" }
```

```toml
# fits/02_evaluate.toml — [method] only
[method]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 500
replicates = 5
starts = { from_mle = "@scout" }
```

```
$ camdl fit run fits/02_scout.toml --seed 1 --label scout
$ camdl fit run fits/02_posterior.toml --seed 1
$ camdl fit run fits/02_evaluate.toml --seed 1
```

TOML ordering bites here: `store` is a top-level key, so it must appear _before_
the first `[table]` header. Written after `[provenance]` it is parsed as a
provenance field and rejected:

```
error: error in fits/02_posterior.toml:
parse error: TOML parse error at line 5, column 1
  |
5 | store = "../camdl-store"
  | ^^^^^
unknown field `store`, expected `derived_from` or `reason`
```

The PGAS file resolves its source first, then runs NUTS on θ given the
conditioned trajectory:

```
  starts source: …/02_scout-77595169/if2-fee126b1/seed_1-e6a47b86 (fit_state.toml)
── method: pgas ──
  chain 0: NUTS enabled (gradient expressions found in IR)
  chain 0: dense mass matrix estimated (sweep 35):
    beta         sd=0.017766
    gamma        sd=0.052177
    correlations: bet-gam=0.42
  chain 0: NUTS fully adapted (sweep 50):
    final step_size: 0.172352
  chain 0 done: 2.7s, acceptance: [beta=92%, gamma=92%]
  chain 1 done: 2.7s, acceptance: [beta=90%, gamma=90%]

Rhat / ESS:
  beta         Rhat=2.065 ✗ ESS=NaN
  gamma        Rhat=1.675 ✗ ESS=NaN
  draws.tsv: 60 posterior samples (all 6 params)

── diagnostics ──────────────────────────────────────
  x parameter acceptance rate 92.5% is outside healthy range [15%, 50%].
  x Rhat for 'beta' is 2.065 (threshold 1.1). Chain estimates have not converged.
  6 error(s), 0 warning(s), 0 info
```

```
── method: pfilter ──
  pfilter rep 1/5: loglik=-58.3
  pfilter rep 2/5: loglik=-58.3
  pfilter rep 3/5: loglik=-58.5
  pfilter rep 4/5: loglik=-58.3
  pfilter rep 5/5: loglik=-58.4

  loglik = -58.4 ± 0.1 (5 reps, 500 particles, 0.2s)
  prequential: elpd=-58.30, mean_crps=316.328, PIT 90% cov=0.08
```

R̂ of 2.07 at 200 sweeps is a not-converged posterior, reported as errors rather
than buried — a real run needs thousands of sweeps. There is a second thing to
read here: `from_mle` is a point rule, so both chains began at the scout's
estimate and the between-chain R̂ cannot say whether the posterior was explored.
`fit summary` on this run prints
`seeded from:  from_mle @scout, run <scout run id>
(every chain at one point)`
and
`R̂ — not assessed: all 2 chains started at
one point (starts = from_mle @scout)`
in place of a verdict (§6.6). To keep R̂ meaningful, start from the priors (drop
`starts`; every parameter here has one) or from a short sampler run's cloud
(`starts = { from_posterior = "@short" }`). The `pfilter` file writes
`prequential.tsv` (one row per scored step: `t`, the joint `log_score`, `ess`; a
step every stream holds as missing is not scored, and a binomial row of `0` of
`0` is missing exactly as `NA` is), `prequential.streams.tsv` (one row per step
and scored stream: `t`, `stream`, `y_obs`, the predictive band
`y_pred_q05`…`y_pred_q95`, `log_score`, `crps`, `pit`) and `prequential.json`.
When the model declares an `origin`, both TSVs carry a `date` column after `t`,
and the surprise table below a `date` column after `t`, each the step's instant
as `--dates` writes it; the JSON is what `camdl compare` reads to score fits
against each other. CRPS and PIT are in the per-stream file only: a step has no
single observed quantity across streams to score them against, and a stream's
`log_score` rows do not sum to the joint one.

Beneath the `prequential:` line, whenever at least one observation was scored,
the filter prints the **surprise table**: the five worst-scored observations of
the trace, worst first (`camdl pfilter --save-prequential` prints the same table
after its own `prequential trace written:` line). The elpd is a sum, and a sum
hides which terms carry it; the table is the answer to "which observations did
the model find hardest?" without opening `prequential.tsv`. From a five-patch
spatial model with two streams bound:

```
worst-scored observations (5 of 53 scored; together 12.3% of elpd -562.3):
         t stream             y_obs      pred q05–q95      log_score   share   pit      ess
       126 cases_p2            1088        [261, 844]   -15.3 (-9.9)    2.7%  0.98    106.1
       287 cases_p2             600        [182, 532]   -13.8 (-8.1)    2.5%  0.96    143.7
       119 cases_p2             411       [426, 1322]   -13.5 (-7.4)    2.4%  0.11    168.8
       238 cases_p2             258         [86, 232]   -13.3 (-7.0)    2.4%  0.97    117.7
       294 cases_p1             579        [178, 568]   -13.2 (-7.5)    2.3%  0.88    156.6
```

Rows are ranked by the step's joint log score; `share` is that score as a
fraction of the elpd, and the header gives the five rows' combined share. A step
has no observed value of its own, so each row names the step's worst stream and
shows _that_ stream's observed value, 5–95% predictive band, (in parentheses)
its own log score and its PIT — the row a reader can look up in the data. `ess`
is the filter's effective sample size at the step (see `filter_ess` under PGAS).
Here the shares are flat at 2–3%, which is the healthy reading: no single
observation dominates. A row carrying a tenth of the elpd by itself, with `pit`
at 0 or 1 and `ess` in the single digits, is one observation the filter could
not explain — check the data row at that `t` before anything else.

Once a sampler run exists, `camdl predict` replays the posterior forward and
writes the predicted-vs-observed pair:

```
$ camdl predict @posterior --n-draws 20
predict: free_forward mode — subsampling 20 of 60 posterior draws (raise with --n-draws)
predict: one_step mode — subsampling 20 of 60 posterior draws (raise with --n-draws)
predict: mode=free_forward+one_step(20 draws) treatment=posterior, 1 scenario(s) [fitted], 1 stream(s), 60 draws from pgas method 'pgas'
wrote camdl-store/<build>/fit/sir-…/runs/pgas-…/seed_1-…/predict/all-…/predictive/weekly_cases.tsv
wrote camdl-store/<build>/fit/sir-…/runs/pgas-…/seed_1-…/predict/all-…/predictive.json
wrote camdl-store/<build>/fit/sir-…/runs/pgas-…/seed_1-…/predict/all-…/observed/weekly_cases.tsv
wrote camdl-store/<build>/fit/sir-…/runs/pgas-…/seed_1-…/predict/all-…/observed.json
```

```
$ cut -f1-7 camdl-store/<build>/fit/sir-…/runs/pgas-…/seed_1-…/predict/all-…/predictive/weekly_cases.tsv | head -3
scenario	time	mode	horizon	treatment	role	quantity
fitted	0	free_forward		posterior	trained	observation
fitted	7	free_forward		posterior	trained	observation
```

(`cut` keeps the excerpt narrow; the full row continues with `fit_rhat_max`,
`fit_ess_min`, the per-row `rhat_mean`/`ess_mean`/`rhat_pred`/`ess_pred`
channels, `n_draws`, and the `q05…q95` band.)

> **Current limitation.** A `pfilter` leaf has no row in `camdl fit table`, and
> no entry in `fit summary --format json`'s `table_rows`; it is listed in
> `leaves`. The text `fit summary` still renders the IF2 and PGAS leaves
> correctly.

### 8.3 Deterministic ODE backend

The same model, fitted against the ODE skeleton rather than the stochastic
process: an NLopt optimiser for the MLE, then gradient-free Metropolis-Hastings
for the posterior. `camdl fit methods` lists every supported (algorithm,
backend) pair; an invalid pair is rejected at load.

This is a **different statistical object** — `p(y | θ, ODE skeleton)`, not
`p(y | θ)` — appropriate for equilibrium or large-population models where the
particle filter is structurally redundant. The runtime says so for every method:

```
ℹ nl-sbplx (Subspace simplex; robust to boundary non-smoothness): deterministic MLE on the ODE-skeleton likelihood.
  camdl computes p(y|θ, ODE_skeleton) under nl-sbplx, not the stochastic-process p(y|θ) IF2/PGAS/PMMH compute. In low-noise regimes the two converge empirically; verify rather than assume.
```

```toml
store = "../camdl-store"

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

[method]
algorithm = "nl-sbplx"
backend = "ode"
chains = 3
max_evals = 200
```

```toml
# fits/05_ode_posterior.toml — [method] only; the rest as 05_ode.toml
[method]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 400
burn_in = 100
starts = { from_mle = "@ode-mle" }
```

```
$ camdl fit run fits/05_ode.toml --seed 1 --label ode-mle
── method: nl-sbplx ──
⚠ Phase 1 typhoid validation passed; other model classes still gathering downstream feedback.

  status: 1 converged, 2 max-eval, 0 failed (of 3)
  chain-agreement: rel range = 49.10% bound | abs range = 3.698e-1   ✗
  loglik-eval:     Δ = 0.0 dB / threshold 30 dB                ✓
  wall: 0.11s (3 chains)

dt-convergence at θ̂: PASS  (|Δ_leg1| = 0.00, |Δ_leg2| = 0.00 nats (vs τ = 0.50); converged.)

$ camdl fit run fits/05_ode_posterior.toml --seed 1 --allow-nonconverged-source
── method: mh ──

ODE marginal-likelihood check at base θ (deterministic)...
  ODE log L = -57.8 (deterministic; no PF variance)
  observations (1 stream):
    ✓ weekly_cases     incidence(infection)         NegBinomial

acceptance rates:
  chain 0: 51.3% ~ high
  chain 1: 51.7% ~ high

Rhat:
  beta         Rhat=1.105 ~ ESS=NaN
  gamma        Rhat=1.051 ✓ ESS=18
```

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

Out-of-sample validation is a key in the config. `holdout_after` under `[data]`
trains on the rows whose period ends at or before the cutoff (§6.2) and leaves
the data file whole:

```toml
store = "../camdl-store"

[model]
camdl = "../models/sir.camdl"

[data]
holdout_after = 56

[data.observations]
weekly_cases = "../data/cases.tsv"

[estimate]
beta = { bounds = [0.05, 1.0] }
gamma = { bounds = [0.01, 0.5] }

[fixed]
from_file = "params/sir_fixed.toml"

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
```

The fit prints the rows it trained on and the first held-out row of each stream
(§6.2) and records its training partition in `fit.meta.json`. The stored problem
keeps the withheld rows' dates and not their values, so their scores need the
file they came from: `camdl predict FIT` and
`camdl score PREDICTION --data
STREAM=FILE` score them (§2.2.4), and
`camdl compare FIT … --data STREAM=FILE` compares them; `--in-sample` compares
the training rows.

A held-out file that already exists on its own, with rows that all end after the
training data, is bound with `[data.holdout]` instead (§6.2).

### 8.5 Comparing and deriving configs

Two read-side commands make a family of fits navigable.

`fit diff` reports what changed between two configs, in fit-semantic terms
rather than as a text diff:

```
$ camdl fit diff fits/01_mle.toml fits/02_posterior.toml
diff: fits/01_mle.toml → fits/02_posterior.toml

Model:
  (no model changes)
Data:
  (no data changes)
Parameters:
  beta: prior (none) → log_normal(mu=-1, sigma=0.5)
  gamma: prior (none) → log_normal(mu=-2, sigma=0.5)

Method:
  algorithm: "if2" → "pgas"
  burn_in: null → 50
  cooling: 0.7 → null
  iterations: 10 → null
  particles: 200 → 50
  starts: null → {"from_mle":"@scout"}
  sweeps: null → 200
```

`fit new` copies a config, stamps a `[provenance]` block onto it, and says how
the new file can start from the source fit's results:

```
$ camdl fit new --from fits/01_mle.toml fits/04_derived.toml
  [provenance] derived_from = "fits/01_mle.toml"
  hint: to warm-start the derived fit from a run of the source, write under its [method]
        starts = { from_posterior = "@<label>" }   # one draw per chain
        starts = { from_mle = "@<label>" }         # every chain at its estimate
        (a handle is @label, a run-id prefix or the run's folder; `camdl list` shows them)
created fits/04_derived.toml
```

`fit table` renders every fit run in a run store, one row each — a problem
holding several runs has several rows, sharing a `fit_id` and told apart by
`run` (the run id's first 12 hex), and every cell in a row is that run's own.
`stem` is the stem of the fit file the run was given:

```
$ camdl fit table camdl-store
fit_id     run          label                  stem           method   methods converged     best_ll ll_type          age
----------------------------------------------------------------------------------------------------------------------------
8bcb3980   d7e6f2f4e33a hold                   fit_pmmh       pmmh     pmmh    no              -38.3 marginal         11s
797c0b53   a372ef8992fa <unlabelled>           fit_pmmh_40    pmmh     pmmh    no              -52.1 marginal         12s
797c0b53   9cd433e2dedb post                   fit_pmmh_all   pmmh     pmmh    no              -53.0 marginal         13s
8bcb3980   603b688f7602 <unlabelled>           fit_seeds      if2      if2     no              -36.7 if2              30s
8bcb3980   556beb126546 <unlabelled>           fit_seeds      if2      if2     no              -37.6 if2              31s
8bcb3980   be6657446294 i                      fit_if2        if2      if2     no              -36.9 if2               4m
```

Labels come from `--label` at run time (§6.8).

Filters: `--converged` / `--gate-failed`, `--with-method if2|pgas|pmmh`,
`--model <hash-prefix>`, `--hash <hash-prefix>`, `--since-seconds <n>`,
`--label-pattern <glob>`. Formats: `text` (default), `json`, `md`, `csv`.
`--quantity <name>` adds a column carrying the posterior median of a scalar
generated quantity, deriving it on demand for a fit that has not been predicted
yet.

Note that `fit_id` is the **problem's** id — model, data, `[estimate]`,
`[fixed]`, settings (§2.2) — with the method and seeds excluded. Two configs
that differ only in `[method]` therefore share a `fit_id`, and `--hash <prefix>`
can return more than one row; the `run` column separates them.

## 9. Provenance and Cache Invalidation

### 9.1 The identity model

Every result is stored in the run store (§4.4), under the id its identity hashes
to; fits are §2.2, simulation cells §3.7 and profile points §2.8. The
human-readable label in each folder name is never identity.

Two rules govern what may enter a hash:

- A field that **changes the stored bytes is identity** and must re-key.
- A pure **re-encoding of the same values is presentation** and is stripped.

A missed re-key silently serves a stale result for different inputs. That is the
worst failure this subsystem can produce, and it is worse than an
over-invalidation.

### 9.2 What enters a run's identity

#### 9.2.1 The encoding

Every stored result's identity is a typed value hashed through one encoder, the
`store` crate's. Its byte rules are the run-store proposal's "Keys" section
(`docs/dev/proposals/2026-11-20-run-store-api.md`), and they are not restated
here. In outline:

- SHA-256, under `HASH_VERSION` (currently `9`), written first at the root, so a
  change to the encoding re-keys every stored result at once.
- A struct or enum writes the tag it states and its version; each field is
  written after its name, in name order; an enum writes its variant's name.
- Every variable-length value writes its length or count first.
- A float enters only as a finite value: `NaN` and `±Inf` are refused, and
  `-0.0` is written as `+0.0`.
- An input file enters by the SHA-256 of its bytes, and an upstream result by
  its run id.

Every field of an identity type is hashed; a field that is not part of the
identity is not a field of it. An identity type that gains a field re-keys every
result of its kind.

#### 9.2.2 The model

A result's model enters its identity as the digest of the stored compiled model
(§3.7): the canonical bytes of every field of the compiled model except its
name, its description, its quantities and its contrasts, the derivative tables
included. Two compiles that differ in anything the engine reads are two models.

`camdl` also computes a structural hash of a model, `ir::model_hash`, which no
identity holds. It is the `model_hash` of a PGAS trajectory manifest (§10.6),
the check that the model given to `predict` or `simulate --fit` is the one the
fit ran on, and `camdl diff`'s "same model" line. It leaves out the model's
name, its quantities and contrasts, and the compiler-derived gradient maps, so a
`--no-state-grad` compile and a full one hash alike.

#### 9.2.3 The identities, per artifact kind

A fit's problem and run identities are in §2.2, a simulation cell's in §3.7 and
a profile point's in §2.8. Each lists its fields; every result's id also holds
the build that computed it (§4.4).

### 9.3 Cache semantics

**`simulate` and the run store.** A `sim` cell is stored in the run store, not
here, and its reuse rule is §4.4: a rerun reuses every stored cell, and
`--force` moves the stored cell to quarantine and stores the recomputed one. An
input changes a cell's id when it changes the cell's identity (§3.7):

| change                                          | new cell? | where it enters                                   |
| ----------------------------------------------- | --------- | ------------------------------------------------- |
| model file edit that changes the compiled model | yes       | model                                             |
| camdl build (engine version, `ir/VERSION`)      | yes       | the build directory                               |
| `--backend`, `--dt`                             | yes       | settings                                          |
| `--output-every`                                | yes       | model[^oe]                                        |
| `--no-flows`, `--columns`                       | yes       | settings                                          |
| `--allow-degenerate-rates`                      | yes       | settings                                          |
| `--param`, `--params`                           | yes       | parameters                                        |
| `--table` file _contents_                       | yes       | model                                             |
| `--integrator`                                  | yes       | model                                             |
| `--scenario`, `--enable`, `--disable`           | yes       | scenario (and parameters, for a `set` or `scale`) |
| `--seed`, `--seeds`, `--replicates`             | yes       | seed                                              |
| `--emit-every`                                  | no[^ee]   | the `obs` by-product                              |
| `--obs*`, `--event-log`, `--quantities-out`     | no        | a by-product                                      |
| `--label`                                       | no        | the invocation record                             |
| `-o`, `--stdout`, `--dates`, `--draws-out`      | no        | exports only                                      |
| `--parallel`                                    | no        | nothing                                           |

[^oe]: `--output-every` enters through the _model_, not the settings:
    `util::rematerialize_with_output_every` rewrites the compiled IR that both
    the engine and the stored compiled model are made from. This is the pattern
    a model-mutating CLI flag must follow **when the flag changes the
    trajectory**.

[^ee]: `--emit-every` deliberately does NOT follow that pattern. It changes only
    the emitted observations, so it enters the identity of the `obs` by-product
    (§10.5) and leaves the cell's id alone: two cadences are two `obs`
    by-products inside one cell. Lowering it into the IR would move the model
    hash a `fit` against real data shares, re-keying a fit over a change
    `emit_schedule` cannot make to a likelihood.

### 9.4 Fit identity and re-running a changed fit

There is no staleness _error_. A fit whose config changed simply resolves to a
different problem or run id, lands in a different folder, and runs; the previous
one stays addressable. There is no stored `config_hash` compared on re-run, and
no `--force` prompt on the fit path.

What each edit does (§2.2 lists what each identity holds):

| edit                                                                                                 | effect                                                      |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| model, training lines of a data file, held-out dates, `[data.tables]` bytes                          | new problem → every run on it re-runs                       |
| `[estimate]` bounds or priors, `[fixed]`, `simplex_groups`, `[config]`, the scenario keys, `ic_free` | new problem → every run on it re-runs                       |
| a held-out value, under `holdout_after` or in a `[data.holdout]` table                               | nothing                                                     |
| a `[method]` field, `starts` and the step tables included, or an `[estimate]` `start`                | new run on the same problem; other method groups are reused |
| a sourced start's source (`from_mle`, `from_posterior`, `from_params`)                               | new dependency → new run for the consumer                   |
| `fit_seeds`, `--seed`                                                                                | new run                                                     |
| the camdl build                                                                                      | new build directory → new problem and runs                  |
| `[provenance]`, comments, `store`, `--label`                                                         | nothing                                                     |

Chain-start _sources_ are keyed by content, not path: a `from_posterior`
source's `draws.tsv`, a `from_mle` source's `fit_state.toml` and a `from_params`
TOML each enter the run's identity as a dependency on that file, so rewriting
the file in place re-keys the run.

**`--resume` is guarded by the run's own identity, less its length.** A resumed
PMMH or PGAS run (the two methods `--resume` accepts) extends its base's chains,
so it is a run of its own beside the base: its identity carries the new length
and a dependency on the base's carried files, every chain's `resume_state.bin`,
`trace.tsv` and `trajectories.tsv`: the chain state the resume continues from
and the rows the resumed run copies, so a base whose rows differ is a different
base. It is not the base's `fit_state.toml`, which records when the base
finished, so two runs of one fixed-seed base address one resumed run.

Continuing a base's chain state is sound only if every other input is the one
the base ran under. The key that checks this is the run's identity with its
length and its base taken out. Each chain's `resume_state.bin` records the key
as `config_hash`, and a resume refuses a chain whose recorded key differs:

```
error: config hash mismatch for chain 0 — cannot resume. Either the
model/data/priors changed since the original run, or this chain predates a
camdl version that changed how a method's identity is computed (…). Re-run
from scratch with --force.
```

Because the key is the run's identity, it covers every input that re-keys a fit
(`[config]` with its `dt`, the scenario, `ic_free`, the held-out dates,
`n_trajectories`) and nothing else.

### 9.5 A fit run's `run_inputs.json`

A fit run is a result in the run store, and its record is the run store's
(§2.5). What the run was asked to do and what it found is its `run_inputs.json`
(§2.2):

```json
{
  "algorithm": { "algorithm": "if2", "backend": "chain_binomial", "chains": 2,
                 "cooling": 0.9, "iterations": 3, "particles": 30 },
  "backend": "chain_binomial",
  "best_chain": 0,
  "best_loglik": -36.852364006324976,
  "binomial": "btpe",
  "chain_starts_kind": "spread",
  "header": { "config": "fit_if2.toml", "method": "if2", "model": "sir.camdl",
              "output": "0.1.0_f531753ac.dirty-a20ab6cb-4bff11f93f55/fit/sir-8bcb39805e9d/runs/if2-e5e94eb7c269/seed_3-be6657446294", … },
  "method": "if2",
  "n_chains": 2,
  "problem_id": "8bcb39805e9d…b0ce",
  "run_id": "be6657446294…b968",
  "seed": 3,
  "starts": "from_prior",
  "starts_source": null,
  …
}
```

`steps` is each estimated parameter's step as the run took it: `kind` (`if2`,
`proposal` or `none`), `map`, each parameter's map tag, and per parameter under
`perturbation` (`if2`) or `proposal` (`pmmh`, `mh`) its `sd_z`, IF2's `at`
(`every_observation` or `time_zero_only`), `from` (`written` or `default`),
`start`, the point θ₀ the step is converted at, and `natural_at_start`, the
natural-scale step at θ₀. `start` and `natural_at_start` are `null` for a
one-sided map with no start and no model value, and for a θ₀ outside the box. An
IF2 run's `simplex` lists each group's members with their `sd_z` and `from`. The
SDs and timings are the method identity's; `from`, `start` and
`natural_at_start` are not keyed.

```json
"steps": {
  "kind": "if2",
  "map": { "beta": "log_interval", "gamma": "log_interval" },
  "perturbation": {
    "beta": { "at": "every_observation", "from": "default",
              "natural_at_start": 0.009309721709264376, "sd_z": 0.05, "start": 0.25 },
    "gamma": { "at": "every_observation", "from": "default",
               "natural_at_start": 0.004736510675075432, "sd_z": 0.05, "start": 0.1 }
  },
  "simplex": []
}
```

`starts_source` is `null` unless `starts` read a source, and then
`{"handle": "@scout", "chains": …, "run_id": "<64-hex>"}`: the handle as
written, the chains a `--starts-chains` selection kept, and the run it resolved
to when this run started (`run_id` is `null` for a file named directly).

`best_chain` is the chain number that produced `best_loglik` — the number of a
`chain_N/` directory in this run, the same value the run's `fit_state.toml`
stores and the method's own stderr line prints, so `chain_<best_chain>/` opens
with no arithmetic. It is `null` for a method with replicates rather than
competing chains (`pfilter`). `camdl show` and `camdl list --format json` print
it as recorded.

**A resumed run.** A `--resume` run is the complete result for the longer chain;
no reader follows the lineage back. Before the new segment runs, each
`chain_N/resume_state.bin`, `chain_N/trace.tsv` and (PGAS, when the base saved
paths) `chain_N/trajectories.tsv` is copied from the base with `std::fs::copy` —
a copy-on-write clone on APFS, a `copy_file_range` (a reflink where the
filesystem supports one) on Linux — and the new segment's rows are appended to
it. The index (`step` for PMMH, `sweep` for PGAS) continues from the base's: the
new segment's first row is one past the base's last, never a repeat of it and
never a restart from 0, and `fit run` refuses a trace whose index does not
increase strictly. Everything a reader takes as the posterior is over the whole
chain: `draws.tsv` (PGAS keyed by sweep, the base's draws first in each chain),
the summary's R̂ and ESS, `trajectories.json`'s `n_draws` and the summary's
saved-path counts, `latent_convergence`, PMMH's acceptance rate and MAP, PGAS's
NUTS divergence counts, and `fit_state.toml`'s best (PGAS takes the better of
the base's best and the new segment's). ESS per second divides by the segments'
summed wall time.

`fit_state.toml`'s `acceptance_rate` is the best chain's over its post-burn-in
steps or retained sweeps. PMMH's covers the whole chain, its accept count
carried in the chain state. PGAS's is the count-weighted combination of the
base's recorded rate, over the chain's retained sweeps below the boundary, and
the new segment's — exact, because both are means over sweeps. The base records
its best chain's rate only, so when the resumed leaf's best chain is another
chain, PGAS's rate covers the new segment's sweeps alone.

Three PGAS outputs are computed from per-sweep sampler internals that the leaf
does not keep for the base's sweeps, and cover the new segment's retained sweeps
only: the `path_renewal` and `filter_ess` summary blocks with `filter_ess.tsv`,
and the conditional-SMC findings in `diagnostics.json`
(`degenerate_ancestor_sampling`, `low_trajectory_renewal`,
`filter_weight_collapse`, `path_renewal_coalesced`, `filter_starved`). The
sampling cost in the resumed leaf's `execution.json` is this execution's.

`run_inputs.json`'s `resume` records the lineage, and is absent on a fresh run:

```json
"resume": {
  "parent_run_id": "f2aea3c8…ad4f",
  "index_column": "step",
  "segments": [
    { "run_id": "f2aea3c8…ad4f", "first_index": [0, 0] },
    { "run_id": "2673d934…e150", "first_index": [60, 60] }
  ]
}
```

`segments` lists every run that contributed rows, oldest first, ending with this
run's own; `first_index` is indexed by chain number. A row of chain `c` whose
index is `i` — `index_column` in `chain_c/trace.tsv`, `draw` in
`trajectories.tsv`, the sweep in PGAS's `draws.tsv` — belongs to the last
segment whose `first_index[c] <= i`, so the last segment's `first_index` is the
resume boundary. A leaf resumed from a resumed leaf copies its parent's segments
forward. How long each segment took is in the resumed leaf's `execution.json`
(`earlier_segments`, then its own `wall_time_seconds`). PMMH's `draws.tsv`
carries no step: draw `k` of chain `c` is the `k`-th retained row of
`chain_c/trace.tsv`, which carries it.

### 9.7 Enumerating a grid — no manifest file

There is no grid-level `manifest.json`. Each completed cell of a `simulate` grid
is independently a `sim` result in the run store (§4.4), and the invocation's
record lists the cells in the order the command asked for them, with their
labels. `camdl cat @LABEL` renders the combined table from that record and the
stored cells; `camdl list` lists the cells; `simulate --dry-run` re-plans the
grid from its flags and reports, per cell, whether the store already holds it.

### 9.8 Identity boundaries worth knowing

Two flags are easy to misjudge, and neither is a gap.

- **`--integrator rk4|rk45` (`simulate --backend ode`)** changes the trajectory
  and changes the cell's compiled model: the override is part of the compiled
  model the cell depends on (§3.7). Two integrators are two cells.
- **`--dates` (`simulate`)** changes only how the combined table is rendered. A
  stored `traj.tsv` never carries date columns; they appear in the `-o`,
  `--stdout` and observation exports only (§10.2). The flag is therefore not
  identity, and a run with and without it stores the same cells.

Related but sound: `reactive_log.tsv` is written whenever the model has an
active reactive policy, so it is a function of the cell's identity and one of
its files; the event log, which only some invocations ask for, is an `event_log`
by-product inside the cell, never one of the cell's files.

---

## 10. Output File Schemas

Every tabular output is tab-separated. The column _roles_ of the files a run
writes are declared in `run.json.output_schema` (§10.9), read back from the
file's own header, so a consumer never has to reverse-engineer a column name.

### 10.1 Trajectories (`traj.tsv`)

The canonical per-cell trajectory file of a `sim` cell. One row per output time.
**No comment or version line** — the first line is the header:

```
time	t_start	t_stop	S	I	R	flow_infection	flow_recovery
0	0	0	990	10	0	0	0
1	0	1	989	9	2	1	2
```

Columns, in this order (`util::TRAJ_TIME_HEADER`, then
`util::TrajColumns::select`):

1. `time` — the output time, rendered with `{}` on an `f64` (`0`, `1`, `0.5`).
   The compartment columns are read **at** this instant.
2. `t_start`, `t_stop` — the half-open period `[t_start, t_stop)` the row's
   `flow_*` columns were accumulated over: `t_stop` is the row's `time`,
   `t_start` the previous row's. The initial-condition row has no interval
   before it and writes the empty period `[time, time)`; its flows are zero.
   When the output schedule has no time at the run's start (an `at = [...]` list
   without it) there is no initial row, and the first row's `t_start` is the
   run's start. A row carries both readings — a state _at_ `time`, a flow _over_
   the period — and names each boundary rather than leaving the period to be
   inferred from the row above.
3. Integer compartments, in model declaration order, rendered `{}`.
4. Real compartments, in model declaration order, rendered `{:.4}`.
5. `flow_<transition>` for every transition, in model declaration order —
   integer flows `{}`, real flows `{:.4}`. Each value is the flow accumulated
   over `[t_start, t_stop)`.

`--no-flows` drops group 5; `--columns A,B,…` restricts groups 3–5 to an
allow-list validated against the model (an unknown name is a hard error listing
the valid ones); the three time columns are always written. Emitted order always
follows the model, never the allow-list. Both knobs are identity (§3.7), so a
filtered trajectory is its own cell.

The stored trajectory never carries a `date` column, regardless of `--dates`.

### 10.2 Combined trajectory table (`-o`, `--stdout`, `camdl cat @LABEL`)

`-o PATH`, `--stdout` and `camdl cat @LABEL` write one table rendered from the
stored cells of an invocation, in the order the command asked for them. It is
not stored; the three write the same bytes.

```
# 0.1.0+<git>
replicate	scenario	seed	time	t_start	t_stop	date	date_start	date_stop	S	I	R	V	flow_infection	flow_recovery
0	(none)	1	0	0	0	2020-01-01	2020-01-01	2020-01-01	9990	10	0	0	0	0
0	(none)	1	1	0	1	2020-01-02	2020-01-01	2020-01-02	9985	13	2	0	5	2
```

- Line 1 is a `# {camdl version}` comment — present here and absent from
  `traj.tsv`.
- The coordinate columns are the same for one cell and for many: `replicate`,
  `scenario`, then `point_id` and one `sweep:<param>` column per swept parameter
  under `--sweep` / `--sweep-file`, or `draw` under `--draws`, then `seed`.
- `replicate` numbers the cells of the whole invocation, 0-based in canonical
  order (scenario, then draw or design point, then replicate), so it runs on
  across scenarios: with 2 draws × 2 replicates, the first scenario's rows carry
  `replicate` 0–3 and the second's 4–7. It is not a key for pairing scenarios; a
  join on it matches nothing across scenarios. Cells of different scenarios at
  the same draw and the same replicate within that draw share a process seed
  (§3.1, the scenario index is absent from the mix), and those are the pairs to
  difference: join on `draw` (or, for a sweep, `point_id`) and `seed` and `t`.
- `scenario` is the arm's label (§2.4): the scenario's name, or for an ad-hoc
  arm `(none)` or the form its flags derive, such as `(none)+vaccination`.
  `seed` is the process seed the cell ran under; passed back to `--seeds` it
  reproduces the cell.
- `date`, `date_start`, `date_stop` appear only under `--dates`, which requires
  the model to declare an `origin`; each is its numeric twin (`time`, `t_start`,
  `t_stop`) rendered via `ir::caltime::internal_to_date_hires`.
- The data columns are identical to §10.1 and honour the same
  `--no-flows`/`--columns` filter.

A table over cells of which one or more stored a failure (§4.4) is refused, and
the error names each such cell.

### 10.3 Posterior draws (`draws.tsv`)

Written by the samplers at the end of a run, into the method leaf. IF2 and the
optimizers write none.

PGAS and PMMH write **all** model parameters — estimated first in `[estimate]`
order, then every remaining model parameter as a constant column — behind two
key columns:

```
chain	draw	beta	gamma	rho	k	N0	I0
0	500	3.41…e-1	1.02…e-1	6.00…e-1	1.00…e1	1.00…e4	1.00…e1
```

- `chain` is the chain's number in every method, matching `PosteriorDraw.chain`
  in `trajectories.tsv` — the join key — and the `chain_N/` directory the
  chain's trace is under (§6.8, one index base).
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

Because PGAS and PMMH carry fixed parameters as constant columns, their file is
a complete parameter specification: a posterior-predictive run needs no
companion `--params`.

### 10.4 Sampler traces (`chain_N/trace.tsv`, `chain_N/parameter_traces.tsv`)

One file per chain, streamed during sampling (header flushed immediately, rows
batched). `N` is the chain's number (§6.8, one index base). A fit run's method
writes them in its working directory outside the store, and they reach the run's
folder when the method ends (§2.2): while the method runs, the run's folder
holds no trace to tail.

PGAS, PMMH and MH write `chain_N/trace.tsv` via the shared
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
sweep	log_complete_data_ll	log_posterior	trajectory_renewal	renewal_b0	renewal_b1	renewal_b2	renewal_b3	renewal_b4	renewal_b5	renewal_b6	renewal_b7	renewal_b8	renewal_b9	collapsed_windows	min_alive	min_ess	min_ess_t	as_opportunity	as_accept	as_proposed	transition_ll	obs_ll	initial_state_ll	obs_ll_cases	obs_ll_confirmations	tree_depth	n_leapfrog	step_size	accept_stat	n_divergent	energy	beta	gamma
```

Three of PGAS's diagnostic blocks need reading before the positions make sense,
and one of them varies in width with the model — so a reader must take the
header as authoritative rather than assume fixed column positions:

- `renewal_b0 … renewal_b9` — always ten columns: `trajectory_renewal` resolved
  in time, one bin per tenth of the substep series. A bin holding no substep
  renders `NA`, not `0.0`: "no substep fell here" and "no substep here was
  renewed" are different diagnoses.
- `collapsed_windows`, `min_alive`, `min_ess`, `min_ess_t` — the sweep's filter
  health, read together with renewal: renewal says whether the path moved, these
  say whether the selection had anything to choose from. `collapsed_windows` is
  how many observation windows left _every_ particle at zero density and
  `min_alive` the fewest particles any window left with a finite weight
  (gh#783). A finite weight can be negligible, so `min_ess` is the smallest
  effective sample size `(Σw)²/Σw²` the weights had at any observation of the
  sweep and `min_ess_t` the observation time it fell at (gh#685) — the number
  the resample after that observation actually drew from. `NA` when the sweep
  scored no observation. A sweep with `min_alive` in the thousands and `min_ess`
  of 3 is the case `min_alive` cannot see: every weight finite, one handful of
  them carrying all the mass.
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

IF2 and the NLopt optimizers write `parameter_traces.tsv` instead, e.g.

```
iteration	loglik	if2_perturbed_loglik	beta	gamma
```

A resumed leaf's `trace.tsv` starts as a copy of its base's and the new segment
is appended under the base's header (§9.5, "A resumed run").

### 10.5 Observation draws

`simulate --obs/--obs-dir/--obs-only/--obs-only-dir` draws synthetic
observations from the model's `observations {}` block. They are an `obs`
by-product of each cell, stored inside the cell folder at
`obs/<stream names joined by _>-<12 hex>/<stream>.tsv`, one file per stream,
beside the by-product's own `identity.json` and `run.json`, and the requested
path or directory is rendered from them.

Per-stream file:

```
time	weekly_cases
0	0
7	10
```

The time column is named `time` here, not `t`. Values render as an integer when
the draw is integral and `{:.6}` otherwise. The stored stream files never carry
a `date` column or coordinate columns; `--dates` and the coordinates reach only
the exports.

Both exports prefix the coordinate columns of §10.2 (`replicate`, `scenario`,
the point columns, `seed`) before the time column, for one cell as for many. The
single-file wide form (`--obs PATH`) then has `time`, an optional `date` under
`--dates`, then one column per stream. It requires every stream to share a
schedule; a multi-cadence model is a hard error directing the user to
`--obs-dir`.

The `obs` by-product's identity is the cell's identity, a digest of the
observation blocks with the `--emit-every` override and the file format, the
streams' names, and the observation seed. A cell already stored without it is
simulated again to draw it, and the trajectory is checked against the stored
one. An observation draw that is refused stores `failure.toml` in the
by-product, and the cell keeps its trajectory (§4.4).

**`--emit-every` (gh#656).** A model's `emit_schedule` is the SIMULATE-only
emission cadence — the fit path reads the data file's own time column and never
consults it — so one model can serve a daily and a weekly emission without
editing its source. `--emit-every N` sets every stream, `--emit-every NAME=N`
one stream by its observation-block label (the IR `source`, so one flag covers a
stratified family); the two forms are mutually exclusive, `N` is a plain number
in the model's `time_unit`, and a stream whose schedule is a fixed `at [...]`
list is refused by name rather than silently converted to a cadence.

On an incidence stream declared with a uniform `covers` form, the override also
re-widens each row's window to the new cadence, keeping the form's anchor — a
daily `closing_at(time, 1 'days)` stream emitted every 7 writes weekly totals
closing at their labels, not one day in seven (gh#833). The file it writes is
then the file of a model declaring that cadence, not of this one;
`simulate
--obs` allows that (nothing has to read the file back), while
`[synthetic]` generation, whose dataset this same model fits, refuses an
override that contradicts the declaration and names the `closing_at` that would
emit it. A stream declared with `window_start`/`window_stop` columns carries its
own boundaries and takes any cadence.

The override is applied at the CONSUMPTION sites — the `--obs*` writers, the
`obs` by-product, an obs-sourced quantity, and `[synthetic]` generation —
**not** by rematerializing the compiled IR the way `--output-every` does. That
is deliberate: `emit_schedule` never enters a likelihood, so moving the model
hash for it would re-key a `fit` against real data over a change that fit cannot
see, orphaning a completed fit. Each path therefore keys what the override
actually determines:

| path                         | what re-keys                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| `simulate --obs*`            | the `obs` by-product's identity, so two cadences are two `obs` by-products inside one stored cell |
| `fit run` with `[synthetic]` | the generated data's own bytes, which the problem's identity hashes — correct, the data changed   |
| `fit run` on real data       | nothing; the flag is refused there, since it could only do nothing                                |

Two cadences share one cell because the trajectory bytes do not depend on the
emission cadence.

### 10.6 Fit sidecars

In the fit problem's folder (§2.2), result files of the problem:

| file                         | content                                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `problem.toml`               | the resolved problem, in `fit.toml` syntax, naming the stored files                       |
| `partition.json`             | which rows of each stream trained                                                         |
| `inputs/<source>.<ext>`      | each data file's training lines                                                           |
| `inputs/tables/<name>.<ext>` | each `[data.tables]` file, as read                                                        |
| `held_out/<stream>.tsv`      | each held-out row's file, position, label and time, without its value                     |
| `model.camdl`                | the model file the fit was given (`model.ir.json` when it was compiled IR), as provenance |

The compiled IR the fit ran against is the compiled-model result the problem
depends on, `<build>/compiled_model/<model stem>-<12 hex>/model.ir.json`.

In each fit run's folder, beside `identity.json` and `run.json`, the files of
§2.2 (`fit.toml`, `method.toml`, `run_inputs.json`, `fit.meta.json`, §2.2.1) and
the method's result files below. The set is method-dependent. `run.json`'s
`files` lists every file with its role: `result` for the method's files and
`method.toml`; `provenance` for `fit.toml`, `fit.meta.json`, `run_inputs.json`,
`execution.json` and an NLopt run's `opt_trace.tsv`; `attempt` for
`progress.json` and `events.jsonl`. Only result files are a function of the
computation, and only they are listed by `camdl show`.

Common to both families:

| file               | content                                                                                                                                                                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit_state.toml`   | θ̂ + run state, with `method` naming the algorithm that wrote it, `best_chain` as the number of the `chain_N/` directory that won, plus `chain_init_source` and `chain_starts_kind`; the artifact a `from_mle` start consumes as a `deps` edge |
| `chain_starts.tsv` | one row per start attempt — the point each chain ran from, and every draw a retry rejected with its `ess` and `reason` — under a header naming the `starts` rule, its kind and the rejected count (§6.6)                                      |
| `progress.json`    | the run's liveness/progress heartbeat, written live, and the terminal state it ends on; every method but `pfilter` writes it (§10.10)                                                                                                         |
| `execution.json`   | the execution record: when this attempt started and finished, its wall time, process, host and threads, and a sampler's cost (§2.2); outside the manifest                                                                                     |
| `events.jsonl`     | what the run reported — its checks, chain progress, findings and end-of-run report — one JSON line per event, as it happened; every method writes it; outside the manifest and the run's identity (§10.11)                                    |

An optimizer leaf (IF2, NLopt) additionally holds:

| file                           | content                                                                                                                                                                                                                                |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mle_params.toml`              | flat `name = value` params plus a `[provenance]` block (camdl version, tamper `content_hash`, `fit_hash`, backend, dt, `model_identity`, per-stream data hashes, seed, method, chain, log-likelihood, `loglik_sd`, `n_particles`, ESS) |
| `final_params.toml`            | the selected parameter vector                                                                                                                                                                                                          |
| `chain_N/final_params.toml`    | per-chain endpoint                                                                                                                                                                                                                     |
| `chain_N/parameter_traces.tsv` | per-chain trace (§10.4)                                                                                                                                                                                                                |
| `chain_evaluations.tsv`        | per-chain loglik evaluation summary                                                                                                                                                                                                    |
| `diagnostics.tsv`              | per-parameter convergence diagnostics                                                                                                                                                                                                  |

A sampler leaf (PGAS, PMMH, MH) additionally holds:

| file                        | content                                                                                                                                                                                                                                                                                                                                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `draws.tsv`                 | the thinned posterior cloud (§10.3)                                                                                                                                                                                                                                                                                                                                                                      |
| `chain_N/trace.tsv`         | per-chain trace (§10.4)                                                                                                                                                                                                                                                                                                                                                                                  |
| `chain_N/resume_state.bin`  | bincode resume state, guarded by its own config hash (§9.4)                                                                                                                                                                                                                                                                                                                                              |
| `chain_N/trajectories.tsv`  | PGAS only — the smoothed latent paths, tidy/long, keyed `chain draw time t_start t_stop [date date_start date_stop]` (§10.1 for the period columns), with a `# camdl-trajectories v1` header line                                                                                                                                                                                                        |
| `chain_N/trajectories.json` | the matching manifest (`format`, `version`, `method`, `granularity`, `n_chains`, `n_draws`, `columns`, `model_hash`, `conditioned`, `calendar`, …)                                                                                                                                                                                                                                                       |
| `latent_convergence.tsv`    | PGAS only, ≥ 2 chains — per (substep, trajectory column): `status` (`constant`/`frozen_disagree`/`mixed`), chain-mean range, R̂ and ESS over the saved paths; binned in `pgas_summary.json`; written at run end, or by `fit summary` from `chain_N/trajectories.tsv` when absent (gh#822)                                                                                                                 |
| `filter_ess.tsv`            | PGAS only — per (chain, observation): mean and minimum filter ESS over the retained post-burn-in sweeps and the sweep count, with a pooled `chain = all` block first; the `filter_ess` block of `pgas_summary.json` carries the summary (particle count, starvation bar, min / 10% / median of the mean profile, starved observations worst first). Omitted when no sweep scored an observation (gh#685) |
| `<algorithm>_summary.json`  | `pgas_summary.json`, `pmmh_summary.json`, `mh_summary.json` — one file per algorithm, deliberately never shared; each names its algorithm under `method` and carries `schema = "camdl.fit-summary/v2"`, which every reader requires                                                                                                                                                                      |
| `diagnostics.json`          | R̂ / ESS / divergence diagnostics, and one `bad_init` record per refused chain — see below                                                                                                                                                                                                                                                                                                                |

**Each parameter's R̂ verdict** (gh#1199) is saved by name under `rhat_band` in
the sampler summary, keyed by parameter like `rhat`, so a reader takes camdl's
verdict instead of re-applying its thresholds. The value is one of `converged`
(R̂ below 1.05), `not_converged` (1.05 up to 1.5), `severe` (1.5 or more, or
`+∞`, which is what every chain frozen at its own value gives), or
`not_assessed` (no R̂ to judge: the estimator refused the parameter, R̂ is
undefined, or the chains all started at one point). It is the band the
end-of-run R̂ table prints as ✓, ~, ✗ or no mark, from the same function, and it
is never greener than an `rhat_high` finding drawn on the same number. The same
field sits beside each parameter's numbers under
`method_result.diagnostics.per_param` in the method's report
(`fit summary
--format json`, and the `method_report` line of `events.jsonl`,
§10.11). camdl's own readers derive the band from the numbers rather than
reading it back.

**A chain that took no step after burn-in** (gh#1032) is one whose retained
draws are all one parameter vector: every estimated parameter has the same value
in every retained draw. Its draws are a point mass at wherever the chain stood
when warm-up ended, and they enter every pooled number. The run records such
chains in the sampler summary under `frozen_chains`, a list of chain numbers
(`[]` when every chain moved; the key is absent in a summary written before
camdl recorded it), and raises one `chain_frozen` finding per chain in
`diagnostics.json`, severity `error`:

| field      | what it holds                                                                                                                                                                                                                     |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chain_id` | the chain's number                                                                                                                                                                                                                |
| `n_draws`  | its retained draws, all equal                                                                                                                                                                                                     |
| `burn_in`  | the warm-up as configured: sweeps for `pgas`, iterations for `pmmh` and `mh`                                                                                                                                                      |
| `n_steps`  | post-burn-in steps in which θ could have moved                                                                                                                                                                                    |
| `sampler`  | tagged by `kernel`: `{"kernel": "nuts", "step_size", "n_divergent"}` — the step size adaptation ended on (`null` if not recorded) and how many post-burn-in steps diverged — or `{"kernel": "random_walk"}`, no proposal accepted |

A frozen chain's `acceptance_rate_unhealthy` finding is not raised, and under
NUTS neither is its `divergent_transitions`: the `chain_frozen` finding states
the stronger fact and carries the divergence count. Any recorded frozen chain
makes the method's verdict `{"kind": "chains_frozen", "chains": [...]}` — not
converged, whatever the R̂, since one frozen chain among moving ones can leave R̂
finite and small. `fit summary --exclude-chains` recomputes the frozen chains
over the retained ones from `draws.tsv` by the same test.

**The per-chain log-likelihood comparison** (gh#406, gh#1008) is computed once,
at the end of the run, and recorded in the leaf; `fit summary`, the warm-start
source check and camdl-scope read the record and do not recompute it. Each
chain's mean log-likelihood is compared with the pooled mean of the other
chains, `Z = |Δ| / se(Δ)` against a Bonferroni cutoff at family-wise α = 0.001,
with backward elimination of the worst chain; a chain with fewer than 10
effective draws, a non-finite mean or no standard error is not assessed. A
sampler compares the retained mean of `log_likelihood` (`pmmh`, `mh`) or
`obs_ll` (`pgas`) with its Monte Carlo standard error, and records it in its
summary under `chain_loglik_agreement`; IF2 compares each start's clean
evaluation (`clean_loglik`) with the standard error of its replicate mean, and
records it in `fit_state.toml` under the same key. The record:

| field                     | what it holds                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `column`                  | what each chain's estimate is a mean of: `log_likelihood`, `obs_ll` or `clean_loglik`                                                                                                                                                                                                                                                                           |
| `column_absent`           | present and `true` only when traces were read and none carries `column`                                                                                                                                                                                                                                                                                         |
| `family_alpha`, `min_ess` | the family-wise α and the fewest effective draws assessed                                                                                                                                                                                                                                                                                                       |
| `chains`                  | one row per chain, ascending: `chain`, `mean`, and `outcome`, tagged by `status`: `{"status": "assessed", "diff", "se", "z", "cutoff", "flagged"}` (`diff` in nats, the chain minus the others) or `{"status": "not_assessed", "reason"}` with `reason` one of `non_finite_mean`, `too_few_effective_draws` (with `ess`), `no_standard_error`, `too_few_chains` |
| `complete_data`           | `pgas` only: `complete` and `transition`, each chain's mean `log_complete_data_ll` and `transition_ll`, parallel to `chains`; shown, never compared                                                                                                                                                                                                             |

The comparison is over every chain the run kept; `fit summary --exclude-chains`
does not recompute it and says so. A sampler's record reaches
`fit summary --format json`, and so `events.jsonl`'s `method_report`, as
`method_result.diagnostics.chain_logliks` (`null` when the leaf does not record
one). Each chain a sampler's comparison flags is also a `chain_loglik_differs`
finding, severity `warning`, carrying `chain_id`, `column`, `diff`, `z` and
`cutoff`. IF2's flagged starts are not findings: the standard error of a clean
evaluation is small enough that starts a tenth of a nat apart differ beyond it,
and whether a spread matters is the decibans leg of the source gate, which has a
floor in log-likelihood units; the comparison names the chains in that refusal.
A leaf written before the comparison was recorded shows it as not recorded.

The `mle_params.toml` `content_hash` is a _tamper_ hash — SHA-256 over
`{name}={value:.12}\0` pairs, truncated to 8 hex — so a hand-edited parameter
value is detectable; `camdl simulate --params` checks it and refuses an edited
file (§2.2). It is not a content hash of the file.

### 10.7 A refused chain's `bad_init` record

When a chain cannot start — under a spread rule, when none of its
`MAX_START_ATTEMPTS = 10` draws could be scored (§6.6); under a point rule, on
its one try — the run skips it and continues (`ran K of N chains` on stderr,
`n_good_chains` in `fit_state.toml`), every attempt is a row of
`chain_starts.tsv`, and `diagnostics.json` carries a `bad_init` record for it:
the chain id (the number the stderr line, `chain_starts.tsv` and `progress.json`
all use), the full estimated-parameter vector it was refused at (so the refusal
is reproducible), a one-line prose `reason` that opens with how many starts were
tried, and an `attempts` array.

`attempts` is the structured half, and it is what downstream tooling should read
— the prose is rendered from the same values, so parsing the sentence gets you
the same facts less reliably. One entry per observation stream scheduled at the
failing observation:

| field                                 | what it holds                                                                                                                                                                |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stream`                              | the declared stream name, not its position in the observation queue                                                                                                          |
| `time`, `date`                        | the observation's model time, and its calendar date when the run is anchored                                                                                                 |
| `cell`                                | `Scored { y_obs }`, `Hole` (scheduled, value missing), or `NotScheduled` (a sibling stream's cadence). All three contribute `0.0` to the joint, and they are different facts |
| `projected_max`, `projected_median`   | the stream's projected quantity across the **live** particles. `null` when no particle was live, or every projection was `NaN`                                               |
| `n_projected_zero`, `n_projected_nan` | how many projections were exactly zero, and how many were `NaN`                                                                                                              |
| `n_neg_inf`, `neg_inf_causes`         | live particles whose per-stream log-density was `-inf`, with the specific guard that fired                                                                                   |
| `n_live`, `n_dead`, `n_particles`     | `n_live + n_dead == n_particles`                                                                                                                                             |

**Dead particles are excluded from every other field**, and this is the point of
the split rather than bookkeeping. A particle killed earlier by a chain-binomial
overshoot carries `-inf` without the observation model having been consulted, so
a mixture of process-model deaths and observation refusals would otherwise read
as a unanimous observation refusal. A refusal with `n_dead` near `n_particles`
is a process-model finding, not an observation one.

Four readings, which have different fixes:

| the numbers                                   | what it means                                                      |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `projected_max` exactly 0, `y_obs` positive   | the model cannot produce this observation here — structural        |
| `projected_max` small against a large `y_obs` | it can; this draw is far out — the prior is the thing to change    |
| `projected_max` finite, `n_neg_inf == n_live` | read `neg_inf_causes`; this is a support violation, not a location |
| `n_dead` near `n_particles`                   | a process-model failure that never reached the observation model   |

Every number in `diagnostics.json`, in a `bad_init` record or any other, is a
JSON number when finite and the string `"inf"`, `"-inf"` or `"NaN"` when not —
an R̂ is `"inf"` when every chain sat at its own single value. `null` means a
field has no value (`projected_max` with no live particle, `param` on a
chain-level acceptance finding), never a non-finite one. A file written before
this encoding carries `null` for a non-finite number, which cannot say which of
the three it was; `fit summary` refuses that entry by name and still reads the
rest, its `bad_init` records included.

### 10.8 Generated quantities

`simulate --quantities-out DIR` writes `DIR/quantities/<name>.tsv` plus a
`DIR/quantities.json` manifest (`schema: "camdl.quantities/v1"`). A single
fixed-parameter run emits a bare `value` per cell (point mode); a
`--draws`/`--replicates`/`--seeds` run emits quantiles over each (point,
scenario) group's replicates.

Each cell's quantities are a `sim_quantities` by-product stored inside the cell
at `sim_quantities/quantities-<12 hex>/quantities.json`, evaluated from the
trajectory at full precision, and the tables are rendered from them. The
quantities block is stored once as a content artifact at
`<build>/quantities/<stem>-<12 hex>/`, and the by-product depends on it; the
cell's own identity does not (`quantities` and `contrasts` are model fields the
compiled model's key leaves out). A cell already stored without its quantities
is simulated again to evaluate them, and its trajectory is checked against the
stored one. A quantity that is not finite stores `failure.toml` in the
by-product, and the cell keeps its trajectory.

Without the flag, a model that declares quantities prints a note and skips them
rather than erroring.

### 10.9 Declared column roles (`run.json.output_schema`)

Rather than reconstruct a writer's column order — which would drift from the
writer — camdl reads each written file's **actual header** and classifies every
column, so the declaration cannot disagree with the file it describes
(`cli/src/output_schema.rs`). The entry is keyed by leaf-relative path, with
`{n}` standing in for the chain index.

`role` (the table's default view) is one of `trajectory`, `observation`,
`posterior_cloud`, `trace`, `predictive`, `landscape`.

Each column's `role` is one of `time`, `window_start`, `window_stop`,
`iteration`, `chain`, `replicate`, `scenario`, `dimension`, `state`, `flow`,
`incidence`, `param_estimated`, `param_fixed`, `observable`, `quantile`,
`diagnostic`. Classification is by reserved name (`chain`, `replicate`,
`scenario`; `t`/`time`/`date` → `time`; `t_start`/`date_start` → `window_start`;
`t_stop`/`date_stop` → `window_stop`;
`sweep`/`step`/`draw`/`iteration`/`point_id` → `iteration`), then prefix
(`flow_`, `inc_`), then membership in the estimated / all-parameter sets, then a
per-table default (`state` for trajectories, `diagnostic` for fit tables).

`time` and `iteration` are deliberately distinct: a trajectory's x-axis is
physical time and a trace's is a sampler index, and conflating them is a real
rendering bug. `window_start`/`window_stop` are not an axis either: they are the
boundaries of the half-open period a row's `flow` columns cover (§10.1), so a
consumer draws a flow as a bar over that span and a state as a point at `time`.

The rendering rule is then mechanical: the x-axis is the `time` or `iteration`
column; group by `chain` / `replicate` / `scenario`; facet by `dimension`;
series are `state` / `param_estimated` / `observable`, with `flow` and
`incidence` spanning `window_start`–`window_stop`; ribbons are `quantile`;
overlays are `diagnostic`. Because the index column is spelled differently by
method (`sweep`, `step`, `draw`, `iteration`), a consumer reads the role and
never the name.

One producer emits a schema today. A completed fit leaf declares `draws.tsv`
(role `posterior_cloud`) plus one entry per per-chain trace filename —
`trace.tsv` for PGAS/PMMH/MH, `parameter_traces.tsv` for IF2/nlopt — read from
the first chain directory with a readable file, so an early crashed chain does
not drop the entry. It is best-effort: an unreadable file is omitted rather than
failing the run.

`profile` and the `camdl predict` outputs declare no schema, and neither do the
results `simulate` and `pfilter` store in the run store. Of the vocabulary
above, the column roles `dimension`, `observable`, and `quantile` and the table
roles `observation`, `predictive`, and `landscape` are defined but unreachable —
no producer emits them.

The schema is recorded, never hashed, so attaching it cannot re-key a run. It is
omitted from `run.json` when empty.

> There is no `camdl summarize` and no automatic per-run summary table (`peak_X`
> / `tpeak_X` / `final_X` / `integral_X`). The `camdl experiment
> summarize`
> subcommand that produced those was removed. Equivalent reductions are
> expressed in the model's `quantities {}` block (§10.8).

### 10.10 `progress.json`: what a method run says while it runs, and how it ends

A background thread rewrites `progress.json` every five seconds — a fixed
wall-clock timer, not a step boundary, because one national-scale PGAS sweep can
take minutes — so it is the one thing a watcher can read while a method runs. In
a fit run's folder the run store writes the method's record inside an envelope,
`{"at": <time written>, "format": 1, "snapshot": {…}}`; the fields below are
those of `snapshot`. Alongside `updated_at`, `pid` and the `state` counter it
carries a `chains` block for any method that reports per-chain liveness:

```json
{
  "updated_at": 1789038341,
  "pid": 2882,
  "state": { "running": { "phase": "burn_in", "step": 276, "total": 2000 } },
  "chains": {
    "total": 24,
    "running": 1,
    "completed": 0,
    "refused": 23,
    "not_started": 0,
    "chains": [
      { "chain": 0, "status": "refused", "reason": "non_finite_start" },
      { "chain": 7, "status": "running" }
    ],
    "numbering": "counted from 0, matching the chain_N/ directories and draws.tsv's chain column"
  }
}
```

A chain is refused at its **start** (§10.7), so how many of the chains a run
paid for are sampling is knowable in its first minutes; `fit_state.toml`'s
`n_good_chains` and the `bad_init` records report the same thing only once the
run has finished. `running + completed` is the number that got past their start
— `total - refused` is not the same thing mid-run, since a chain that has not
started is neither good nor bad.

**A chain with no row has not started.** That is the distinction an empty
`chain_N/` directory cannot make: a refused chain and one queued behind
`--parallel` call for opposite responses (respecify, or wait), so a chain that
has not been picked up by a worker is counted in `not_started` and reports
nothing. The block is **absent**, not null, for a method that does not report
per-chain liveness — today every method but PGAS — and `chain` here is the same
number `chain_starts.tsv`'s `chain_id` column, the `bad_init` records and the
`chain_N/` directories use, so joining any two of them takes no arithmetic.

**An NLopt fit adds one row per search.** `step` is the furthest any search has
reached, which on a multistart fit says nothing about the slowest search or
whether any of them is still improving. While an `nl-sbplx`, `nl-bobyqa` or
`nl-lbfgs` fit runs, `progress.json` carries a top-level `searches` array beside
the unchanged `state` counter:

```json
{
  "updated_at": 1794981175,
  "pid": 6919,
  "state": { "running": { "phase": "optimizing", "step": 11, "total": 5000 } },
  "searches": [
    { "chain": 0, "evals": 11, "best_loglik": -56.56391717561852 },
    { "chain": 1, "evals": 10, "best_loglik": -49.81050630695637 }
  ]
}
```

`chain` is the search's number in `chain_results.tsv` and `opt_trace.tsv`;
`evals` is its evaluation count and `best_loglik` the best finite log-likelihood
it has scored, `null` until it scores one. A search's row is updated each time
it writes a row of `opt_trace.tsv` (§10.13), so the two agree and the row is at
most 25 evaluations or 2 seconds behind the search, and again with its final
count when it ends; the terminal record keeps the block. A search with no row
has not made its first evaluation. The array is absent, not null, for every
other method.

#### Which methods write it, and what a step is

Every method writes `progress.json` except `pfilter`, whose unit of work is a
replicate of one fixed point — neither an MCMC phase nor a search, so it
publishes no counter rather than one labelled with a phase it is not in. For the
rest, `step` of `total` counts:

| method                              | one step is              | `total`      | `phase`                             |
| ----------------------------------- | ------------------------ | ------------ | ----------------------------------- |
| `pgas`                              | one sweep                | `sweeps`     | `burn_in` → `sampling` at `burn_in` |
| `pmmh`, `mh`                        | one MCMC iteration       | `iterations` | `burn_in` → `sampling` at `burn_in` |
| `if2`                               | one cooling iteration    | `iterations` | `optimizing`                        |
| `nl-sbplx`, `nl-bobyqa`, `nl-lbfgs` | one objective evaluation | `max_evals`  | `optimizing`                        |

`step` is the furthest **any** chain has reached, and it never moves backward.

A `camdl profile` invocation writes the same file, in phase `profiling`, into
its own directory under the profile's base rather than into a leaf: there a step
is one job (a grid cell's start at one seed) finished, a cached one included, of
the invocation's `total_jobs` (§10.12).

#### The terminal state

A method run that finishes writes `"state": "done"` after every other output in
the leaf is on disk, so `done` is the cue that the final statistics are ready to
read. A run that camdl stops writes `"state": {"failed": {"reason": …}}`,
carrying the same sentence the command printed on stderr — so a reader holding
only `progress.json` learns what a reader of the terminal learns. Both are
written before the leaf is finalized.

A `running` record that has stopped being refreshed means neither: it is what a
run leaves when it is SIGKILLed or panics, and **deadness is the reader's
inference from staleness, never a self-claim** — a killed process cannot write
"I died". Treat a `running` record older than a few refresh intervals as
presumed dead, and a `failed` record as a failure camdl saw and explained.

### 10.11 `events.jsonl`: what a method run reported, as values

The terminal shows a method run's checks, progress, findings and end-of-run
block as text. `events.jsonl` records the same values, one JSON object per line
in the order the run reported them, so a reader that is not the terminal — camdl
'scope, a terminal UI — reads values and never parses the text. Every method
writes it, `pfilter` included, from the moment the method begins.

**Every line** carries four fields beside the event's own:

| field     | what it holds                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------------- |
| `schema`  | `"camdl.events/v1"`. A changed meaning or a removed field is a new version; a new event or field is not |
| `seq`     | the line's place in the file, from 0                                                                    |
| `elapsed` | seconds since the method began                                                                          |
| `event`   | which event: `begin`, `check`, `progress`, `finding` or `method_report`                                 |

**The events:**

| `event`         | when                                                                              | its fields                                                                                                                                                                                                                                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `begin`         | first, as the method starts                                                       | `method`; `header`, the run's opening block (`config`, `method`, `model`, `estimate`, `fixed`, `starts`, `output`), which no other file in the leaf holds until the leaf commits; `acceptance`, the healthy band the method's acceptance diagnostics apply (`kernel`, `band_lo`, `band_hi`), or `null` for a method without them           |
| `check`         | as each check before sampling is shown                                            | `check`, tagged by `kind`: `obs_stream` (`name`, `carrier`, `family`), `filter_noise` (`check`: `n_particles`, `pairs`, `sigma`, `s`, `ll_mean`), `method_caveat` (`method`, `note`, `series_len`), `starts` (`mode`, `params`, `note`)                                                                                                    |
| `progress`      | per chain, at its first step, then at most every 30 seconds, and at its last step | `chain`; `step` of `of`, the chain's progress bar's count, which is the chain's own step (a chain resumed from step 2000 of 2500 reports 2001 first); `loglik` (`value`, and `kind`: `marginal`, `if2`, `ode_marginal` or `complete_data` — PGAS's, not comparable with the others) or `null`; `accept` or `null`                          |
| `finding`       | as each finding is pushed                                                         | `finding`, the record `diagnostics.json` holds for it (`kind`, tagged by `type`; `severity`; `message`; `method`)                                                                                                                                                                                                                          |
| `method_report` | last, as the end-of-run block is shown                                            | `outcome`: `completed`, with `report`, the method's report: the object `fit summary --format json` gives for it under `methods`, built from the run's memory rather than its files, with non-finite numbers spelled as that document spells them (`"NaN"`, `"inf"`, `"-inf"`); or `failed`, with `error`, the sentence the command printed |

`progress` is written whatever `--progress` says; the display mode governs only
the terminal. A step is what `progress.json`'s step counts for the method
(§10.10). PGAS, PMMH, `mh` and IF2 chains report progress; NLopt and `pfilter`
report none.

**The acceptance band.** `acceptance` on `begin` is
`{"kernel": "random_walk", "band_lo": 0.15, "band_hi": 0.5}` for `pmmh` and
`mh`, whose θ-move is random-walk Metropolis–Hastings; for `pgas` it is
`{"kernel": "nuts", "band_lo": 0.6, "band_hi": 0.95}` when the sampler runs NUTS
on the model (it needs the compiler's rate gradients) and the random-walk band
when it falls back to Metropolis-within-Gibbs; it is `null` for `if2`, the NLopt
methods and `pfilter`. The bounds are fractions, inclusive at both ends. An
`acceptance_rate_unhealthy` finding carries the same `kernel`, `band_lo` and
`band_hi` beside its `rate`, so a reader has the band whether or not a finding
fires and never parses it from the message, which is rendered from those fields.
An entry written before the band was recorded has neither field: it is read, its
band is not recorded (no number is assumed for it), and `fit summary` prints the
message it stored, as written.

A finding's line is written when the run pushes the finding: a refused chain's
`bad_init` as the chain is refused, and a sampler's per-chain and R̂ findings
after its chains join, in chain order, after the last chain's `progress`.
Numbers follow §10.7's rule: a non-finite number is the string `"inf"`, `"-inf"`
or `"NaN"`, and `null` means no value.

**How a log ends.** A run camdl finished or stopped ends its log with
`method_report`, whichever method failed and wherever, its findings recorded in
`diagnostics.json` beside it. A log whose last line is something else was cut
off: the process was killed or panicked, and `progress.json` says which
(§10.10).

**Writing.** Each line is written whole, in one write, so a reader following the
file sees whole lines; the file is not synced to disk line by line. A line that
cannot be written fails the run, as an unwritten `diagnostics.json` does (exit
status 1, `error: cannot write …/events.jsonl: …`): before the method is
recorded `done` if the failure came while it ran, after the end-of-run block if
it came at the report, the leaf then already committed.

**Outside the stored run and the identity.** The log is written in the method's
working directory while the method runs, and `fit run` does not store it with
the run. It enters no hash: it is a record of what one execution showed, not an
input to anything a run computes. In consequence:

- **A reused run writes nothing.** The method does not run, and the line on
  stderr that says the run was reused is not an event.
- **A resumed run** writes its own log in its new leaf; the base leaf's log is
  not copied or touched.
- **A recomputed leaf** (`--force`, or a stale `Running` leaf reclaimed after a
  killed run) starts a new log: the reclaim clears the dead run's log with its
  other files, and the log is created empty when the method begins.

### 10.12 A profile's invocation: what it will compute, and how far it has got

A profile's points are claimed together when the command starts, and each is
visible in its folder, with status `running`, while it computes (§2.8). The
invocation record (`<root>/.invocations/<seq>.json`) lists every point the
command asked for, in the order (seed, cell, start), each with its attributes
(`model`, `focal`, `method`, `seed`, `start`) and its outcome: `computed`,
`reused`, `failed` with its reason, or `busy`. While the command runs,
`<root>/.invocations/<seq>.progress.json` holds a snapshot with phase
`profiling`, the points finished (stored ones included) and the total. The
profile table is rendered from this record and the stored points, so `--label L`
and `@L` name the profile as a whole.

### 10.13 `opt_trace.tsv`: each NLopt search's running best, while it runs

An NLopt search reports nothing in `chain_results.tsv` until it finishes. On a
model whose evaluation costs 0.4 s, an 8,000-evaluation search is close to an
hour with no output, and a search that stopped at `max_evals` cannot be told
apart from one that had levelled off. `opt_trace.tsv` records each search's
climb as it happens. An `nl-sbplx`, `nl-bobyqa` or `nl-lbfgs` leaf holds one,
beside `chain_results.tsv`, with a row from every search:

```tsv
chain	eval	loglik	best_loglik	beta	gamma
2	1	-206.26724624340633	-206.26724624340633	0.6177941455696034	0.28345867819301973
1	1	-103.87751305001734	-103.87751305001734	0.1979316379838952	0.036382023253224934
0	1	-4763.2417810710185	-4763.2417810710185	0.05880848536922149	0.04171466478911118
```

| column           | what it holds                                                                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chain`          | the search, numbered as in `chain_results.tsv`                                                                                                      |
| `eval`           | the search's evaluation count at this row, from 1                                                                                                   |
| `loglik`         | the ODE log-likelihood this evaluation scored, as the objective returned it; `-inf`, `inf` or `nan` when it is not finite                           |
| `best_loglik`    | the best finite log-likelihood the search has scored so far; empty until it scores one                                                              |
| one per estimate | the point that scored `best_loglik`, on the natural scale, in `chain_results.tsv`'s column order and spelling; empty until a finite value is scored |

**The header is written before any search starts.** Rows follow in the order
they are written, so the searches' rows interleave; group by `chain` to read one
search. Within a search `eval` increases and `best_loglik` never decreases.

**A row at the first evaluation, then at least every 25 evaluations or every 2
seconds, whichever comes first, and one at the search's last evaluation.** The
25-evaluation bound keeps a fast search (milliseconds per evaluation, thousands
of evaluations) to a few hundred rows; the 2-second bound keeps a slow one (0.4
s per evaluation) moving in a live plot. The last row is at the count
`chain_results.tsv` reports as `n_evals`, and its `best_loglik` and point are
that search's result: equal to `chain_results.tsv`'s `loglik` and parameters at
the precision that file prints. For `nl-lbfgs` NLopt reports its final iterate
rather than the best point it evaluated, so the trace's last `best_loglik` can
exceed the reported `loglik` in the last few digits.

**Each row is one whole-line write to a file opened for appending**, so a reader
tailing the file while the searches run sees whole rows only, and rows from
searches running in parallel never interleave within a line.

**A record of the attempt, outside the identity.** Which evaluations get a row
depends on how fast this execution ran, so two executions of one fit can write
different traces. `opt_trace.tsv` is therefore one of the records of the attempt
(§2.2): outside the leaf's manifest and every hash, and moved into
`.attempts/<n>/` with the other records when a dead attempt's leaf is reclaimed.
The search's result is `chain_results.tsv`, which is in the manifest. The trace
only reads the value the objective returns; the optimiser's path and result are
the same with or without it. A `camdl profile` runs the same optimiser in each
cell and writes no trace.

## 11. Predictive Workflows

Every workflow in this section is a `camdl simulate` run over a **cloud of
parameter vectors** (`--draws`) rather than a single point. The four sources are
`prior`, `posterior`, `uniform`, and a path to a parameter TSV. A run with
`--draws`, `--replicates`, or `--seeds` is an _ensemble_: one stored cell per
grid point, which the invocation's record lists in order.

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

**Model-only form: every parameter needs a prior _or_ a value in every
scenario.** The draws carry only the parameters with a prior; every other
parameter resolves per cell from `--params`, that cell's scenario, or `--param`.
`--scenario a,b` is two sets of cells, as on every other `simulate` path, and
each resolves against its own scenario — a value one scenario sets never reaches
the other. A parameter left unset in some scenario is refused before any cell
runs, naming the scenario:

```
$ camdl simulate v2.camdl --draws prior -n 3 --scenario half,other
error: parameter 'c' has no prior, and nothing gives it a value for scenario 'other'.
  --draws prior samples only the parameters that have a prior; every other parameter takes its value from --params FILE, the scenario's `set`, or --param NAME=VALUE.
  Fix options: declare a prior on 'c' in the model's parameters {} block (`c : <type> in [lo, hi] ~ <dist>(...)`), supply `--fit FIT.toml`, set it with --params FILE or --param c=VALUE, or set it in scenario 'other'.
```

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
TSV with one row per draw and one column per _sampled_ parameter. Parameters the
measure does not vary have no column, so the file depends only on the measure
and the seed — not on `--params` or the scenario order — and it round-trips
through `--draws <PATH>` given the same `--params` and scenarios:

```
$ camdl simulate v2.camdl --draws prior -n 3 --params b0.toml --draws-out p.tsv
$ head -2 p.tsv
beta	gamma
2.90465033202471634e-1	1.63870215243367284e-1
```

Prior draws are rejection-sampled against each parameter's declared bounds; the
run reports per-parameter rejection counts when truncation was active.

### 11.2 Posterior Predictive Check

_"Does my fitted model generate data that looks like the real data?"_

Prefer `--draws posterior --fit <fit run>`, which resolves the run's canonical
post-warm-up cloud for you rather than making you find the file:

```bash
camdl simulate models/sir.camdl \
    --draws posterior --fit @posterior \
    --replicates 10 --obs-dir ppc/
```

It prints what it resolved on a `draws: posterior` line: the number of draws,
the method that wrote them, and the `draws.tsv` it read.

Resolution is **by artifact, not by method name**: a run has a posterior iff it
wrote `draws.tsv`. An optimizer-only fit (IF2, `nl-sbplx`, `nl-bobyqa`,
`nl-lbfgs`) resolves to an error rather than dressing a single point up as a
distribution:

```
error: --fit @mle: the run holds no posterior draws (draws.tsv); a sampler (pgas, pmmh, mh) writes them
```

The canonical file is the run's `draws.tsv` — post-warm-up and thinned — **not**
`trace.tsv`, which carries warm-up rows for live observability.

A raw TSV path still works. If the file carries only the estimated columns (a
posterior trace tail), pass `--fit` alongside it and the fit's `[fixed]` block
backfills the missing parameters — never overwriting a column the file provides:

```bash
camdl simulate models/sir.camdl --draws draws.tsv --fit fit.toml \
    --replicates 10 --obs ppc.tsv
```

**The purpose-built alternative.** `camdl predict` writes the
predicted-vs-observed artifact directly from a completed fit, banded over the
posterior, with no re-plumbing of draws:

```bash
camdl predict @posterior
```

It emits, under the run's `predict/<mode>-<12 hex>/` (§2.2.2):

```
predictive/<stream>.tsv   time | <dims…> | mode | horizon | treatment | role | quantity
                          | fit_rhat_max | fit_ess_min
                          | rhat_mean | ess_mean | rhat_pred | ess_pred
                          | n_draws | q05..q95
observed/<stream>.tsv     time | <dims…> | value
```

with two modes where the backend supports them — `free_forward` (one forward
replay per draw) and `one_step` (the filter's one-step-ahead predictive; chain-
binomial only, a hard error on an ODE fit). The posterior cloud is subsampled to
`--n-draws` (default 200) by a strided pick across the whole cloud.

### 11.3 Scenario Prediction Under Posterior Uncertainty

_"What would happen under an SIA, given what we learned from the data?"_

```bash
camdl simulate models/sir.camdl \
    --draws posterior --fit @posterior \
    --scenario baseline,with_sia \
    --replicates 10 --obs-dir obs/
```

`--scenario a,b` is a **partition of the cell grid**: each scenario is its own
set of runs, not a layering of one scenario on another. For each (draw, seed)
pair both scenarios are simulated with the same seed, giving paired comparisons
that propagate posterior uncertainty.

**Every arm starts from the model that was fitted.** A fit config's top-level
`scenario` or `enable`/`disable` (§6.2) is part of the problem the posterior
describes: `fit run` estimated under it. Every command that replays a fit
applies it — `simulate --draws posterior --fit`, `camdl predict` (the `fitted`
arm, every `--scenario` arm, the one-step filter), the contrast arms,
`pfilter
--fit` and the prequential `compare` derives. For a stored fit the
selection is read from its `problem.toml`; the problem's identity already holds
it (§2.2).

On the replaying commands each arm's own changes apply **on top of** the fit's
selection, the way a scenario `compose`s a sub-scenario: the fit's
`enable`/`disable` lists first, the arm's after, and an explicit disable wins.
So on a fit with `enable = ["lockdown"]`, `fitted` runs `lockdown`, an arm
`with_sia { enable = [sia] }` runs `lockdown` and `sia`, and an arm
`no_lockdown { disable = [lockdown] }` runs neither; an arm that toggles nothing
is the fitted model. The command says so on stderr when the fit selected
anything:

```
predict: the fitted model runs the fit's own enable ["lockdown"] / disable []; every scenario arm applies its changes on top of it
```

Only the fit's intervention lists form the base. A fit scenario's `set`/`scale`
is already in the draws: `fit run` refuses a scenario that moves a `[fixed]`
value or touches an `[estimate]` parameter (§1.3), so applying it again could
only re-apply a value the draws carry. On `simulate` the base enters each cell's
`scenario` identity level (its lists lead the arm's), so a cell replayed under a
fit's selection never shares a `run_id` with the same draw and seed replayed
without it.

**The coupling is paired-seed CRN, not event-keyed RNG.** The runtime uses a
stateful ChaCha8 PRNG, so two arms agree only while they consume draws in the
same order. `enable`/`disable` scenarios are byte-identical up to the first
state divergence; `set`/`scale` scenarios that perturb propensities from `t=0`
are correlated but never identical. Any structural change that reorders draws
also breaks the coupling.

**Do not hand-difference the two arms for a headline number.** The model-level
`contrasts { }` block does this correctly and is auto-emitted by
`camdl predict`:

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
the last saved trajectory snapshot strictly before the instant it fires — on the
arms' chain-binomial grid `t_start + k·dt`, not its declared time (declared at
2.4 with `dt = 1`, it fires at 2) — so "fork at or after the intervention" is
unrepresentable. The resumed arms do not re-apply an effect at the fork time:
the saved state is already post-effect. Both arms share the smoothed latent
state at the fork and the per-draw seed, then desync by design after it. Results
are banded over the forkable draws into `contrasts/<name>.tsv`. A contrast with
no toggled intervention (a parameter-only scenario) or one toggled by a
parametric or reactive fire time is skipped with a note, never silently.
`contrasts.json` lists each table written (`contrasts[].name` and
`contrasts[].file`), and none for a contrast skipped; a table it does not list,
or one it lists that was not written, is refused when the prediction is stored.

**A chain selection reaches the contrast.** `camdl predict --exclude-chains`
narrows the forkable set to the retained chains before the fork, so a contrast
bands over exactly the cloud `predictive.json`'s `chain_selection` block
describes — the contrast's `n_used` and the free-forward rows' `n_draws` are one
number. The retained-chain scope is printed alongside the count. Excluding
chains narrows an already-partial set (only path-saved draws are forkable); if
the intersection is empty the run is refused by name, never emitted as an empty
band.

**`--quantities-out` bands each scenario separately.** `quantities { }` bands
are taken over draws × replicates × seeds; scenario is a partition of that grid,
not part of the band. A multi-scenario run therefore writes one band per
scenario, never a pooled one: each `quantities/<name>.tsv` carries a leading
`scenario` column (`scenario  n_draws  q05  q25  q50  q75  q95`, one row per
scenario) and `quantities.json` lists one entry per quantity and scenario.

### 11.4 Uniform Exploration

_"What does the model do across parameter space?"_

```bash
camdl simulate models/sir.camdl --draws uniform -n 500 --replicates 1
```

Samples uniformly from each parameter's declared bounds. No Bayesian pretension
— this is space-filling exploration for model debugging.

`camdl design lhs MODEL --fit fit.toml --n 200 --seed 1 [-o design.tsv]` writes
a Latin-hypercube design over a fit config's `[estimate]` bounds, a TSV with
`point_id` and one column per estimated parameter, which
`camdl simulate MODEL --sweep-file design.tsv` runs point by point and
`camdl pfilter MODEL --fit fit.toml --sweep-file design.tsv` scores point by
point (§5.5, "Scoring a design"). `camdl survey` is refused as an unrecognized
subcommand.

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

`camdl fit run` then generates one dataset per sim seed into
`<fit_dir>/synthetic/data/ds_NN/` — one file per observation stream, under the
columns that stream declared, exactly as `simulate --obs-dir` writes them — and
runs the `[method]` once per dataset. Each dataset is bound and read back
through the loader real data uses, so a windowed stream (one declaring
`window_start`/`window_stop` columns) generates and fits like any other. Each
grid cell is its own content-addressed fit, readable with `camdl list` / `show`
/ `cat`.

Two honest limitations:

- This is **calibration at a fixed truth**, not simulation-based calibration.
  Classical SBC (Talts et al. 2018) draws θ from the prior _per dataset_ and
  checks rank uniformity; `[synthetic]` holds `true_params` fixed across every
  dataset and asks whether the estimator recovers that one point.
- The cross-cell roll-up — the parameter-recovery coverage and bias table — is
  **not built**. The run prints
  ``note: grid summary / coverage are derived views — owed by `fit recovery`
  (gh#150 / gh#154)``.
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

Two chains, resolved by the same code (`resolve_prior` in `fit/runner.rs`) but
with a **different tier-3**. The difference is deliberate and load-bearing.

`camdl fit run`, sampler methods (`pgas`, `pmmh`, `mh`):

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
`camdl predict`, `compare`), so silently targeting the unconditioned likelihood
is not allowed. The refusal names every parameter and all three remedies:

```
$ camdl fit run fit_noprior.toml --seed 1
error: [method] (algorithm = "pgas") has parameters with no resolved prior:

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

`camdl profile` resolves the same first three tiers, then falls back to
`Prior::Flat` (source `flat_fallback`) without a warning: every profile cell is
a maximization (`if2` or an `nl-*` optimizer) that reads no prior, so the
resolution is provenance only. It is written to the profile-base sidecar and
keys the profile's base identity. `--starts from_prior` draws starting points
from the resolved priors.

Per-parameter prior provenance is written to `fit.meta.json` under
`resolved_priors` and under `partition.parameters.<p>.prior`, for every method
(sources `fit_toml`, `model_ir`, `flat_explicit`, and `flat_fallback` on a
problem no sampler can run on). A profile writes `resolved_priors` to its
profile-base sidecar.

### 12.2 When priors are used at all

| Method algorithm                    | Prior used?                                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------ |
| `pgas`, `pmmh`, `mh`                | Yes — the prior density enters the acceptance ratio; a missing prior is a hard error |
| `if2`                               | No — iterated filtering is maximum likelihood; the code never reads a prior          |
| `nl-sbplx`, `nl-bobyqa`, `nl-lbfgs` | No — deterministic MLE                                                               |
| `pfilter`                           | No — likelihood evaluation only                                                      |

`--draws prior` needs a proper prior on every _sampled_ parameter; every other
parameter is left out of the draws and resolves per cell from the fit's
`[fixed]`, `--params`, the scenario, or `--param` (§11.1).

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

### 12.4 Prior support and the search box

There is no `transform` key in `[estimate.<p>]` (a file carrying one is refused
at load as an unknown field). The transform follows from the parameter's
resolved bounds and declared type (§6.3), and every prior density reads the
logarithms it needs from the transformed coordinate, so the prior is exact under
whichever transform results. What is checked, before the fit starts, is that the
prior puts mass over the whole search box: a prior that is zero on part of the
box would be a wall by another route.

| Prior family                                        | The resolved search box must lie within |
| --------------------------------------------------- | --------------------------------------- |
| `log_normal`, `half_normal`, `gamma`, `exponential` | `[0, ∞)`                                |
| `beta`                                              | `[0, 1]`                                |
| `uniform(L, U)`, `log_uniform(L, U)`                | `[L, U]`                                |
| `normal`, `flat`                                    | any box                                 |
| `truncated_normal`                                  | equal to its truncation                 |

The error names the parameter, the prior's support, the search box that reaches
outside it, the two fixes (narrow `bounds` in `[estimate.<p>]`, or choose a
prior whose support covers the box), and the resolved prior's source
(`validate_prior_transform_compat`).

---

## 13. Utility Commands

### 13.1 `camdl fit table`

Walks a run store and renders one row per fit run — its run id, method,
convergence verdict, best log-likelihood and its type, age. Rows of one problem
share a `fit_id`; the JSON row carries the full `run_id` (table_row schema v4,
in which a non-finite number is the string `"NaN"`, `"inf"` or `"-inf"`), the
CSV appends it as the last fixed column. The optional `ROOT` argument names the
run store's root; without it `fit table` reads `$CAMDL_STORE`, else
`./camdl-store`.

```
$ camdl fit table camdl-store
fit_id     run          label                  stem           method   methods converged     best_ll ll_type          age
----------------------------------------------------------------------------------------------------------------------------
8bcb3980   d7e6f2f4e33a hold                   fit_pmmh       pmmh     pmmh    no              -38.3 marginal         11s
797c0b53   a372ef8992fa <unlabelled>           fit_pmmh_40    pmmh     pmmh    no              -52.1 marginal         12s
797c0b53   9cd433e2dedb post                   fit_pmmh_all   pmmh     pmmh    no              -53.0 marginal         13s
8bcb3980   603b688f7602 <unlabelled>           fit_seeds      if2      if2     no              -36.7 if2              30s
8bcb3980   556beb126546 <unlabelled>           fit_seeds      if2      if2     no              -37.6 if2              31s
8bcb3980   be6657446294 i                      fit_if2        if2      if2     no              -36.9 if2               4m
```

Read-only by default: every cell is recovered from the stored run's files. **One
flag breaks that rule** — `--quantity <NAME>` may _derive_ a value on demand,
running `camdl predict --mode free_forward` for any fit run that has not been
predicted yet, and storing that prediction, with its `quantities-model/` tables,
inside the run. Optimizer fits have no posterior cloud, so their cell renders
`—`.

Filters: `--converged`, `--gate-failed`, `--with-method if2|pgas|pmmh`,
`--model <HASH_PREFIX>`, `--hash <HASH_PREFIX>`, `--since-seconds <N>`,
`--label-pattern <GLOB>`. `--baseline <HASH_PREFIX>` picks the fit that the Δ
columns are measured against (default: lowest hash in the surviving cohort).
`--exclude-chains <IDS>` applies only to derived `--quantity` cells and always
warns, because post-hoc chain exclusion biases the posterior toward the retained
mode.

`--format text|json|md|csv`. The JSON document is schema-pinned; each row
carries considerably more than the text view:

```
schema, fit_id, fit_hash, label, stem, model_identity, methods, method,
config_diff_from_baseline, converged, gate_verdict, best_loglik, loglik_type,
max_chain_agreement, max_rhat, acceptance_rate, ess_at_mle, ess_posterior,
ess_per_iter, ess_per_sec, params, delta_ll_vs_best, age_seconds, created_at,
stale, stale_reason
```

`gate_verdict` is the method's convergence verdict, the one `fit summary` and a
`from_mle` / `from_posterior` source check read:

| value                  | method    | meaning                                                                                                                                                                                                                                                                                                                                        |
| ---------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pass`                 | if2, nl-* | every leg passes                                                                                                                                                                                                                                                                                                                               |
| `fail_a`               | if2       | some estimated parameter's tail chain agreement Â (initial-state ones included) is at or above the threshold                                                                                                                                                                                                                                   |
| `fail_db`              | if2, nl-* | the chains' log-likelihoods spread past the decibans floor                                                                                                                                                                                                                                                                                     |
| `fail_both`            | if2, nl-* | both                                                                                                                                                                                                                                                                                                                                           |
| `incomplete`           | if2       | no leg failed, but some Â is not finite (within-chain variance collapsed), or the run lost started chains (`n_good_chains < n_chains`; `chain_starts.tsv` marks them `skipped`), so the legs judged the survivors only: not a pass, and a starts-source check refuses it                                                                       |
| `fail_chain_agreement` | nl-*      | the converged chains' estimates disagree                                                                                                                                                                                                                                                                                                       |
| `fail_status`          | nl-*      | a chain ran out of evaluations or failed                                                                                                                                                                                                                                                                                                       |
| `not_assessed`         | if2, nl-* | nothing was assessed: the run asked for one chain (one converged nl-* chain), so there is nothing to compare between chains; or an nl-* leaf that predates the recorded optimizer verdict (re-run it with `--force`). Not a pass, though a starts-source check accepts it. A multi-chain IF2 run left with one chain is `incomplete`, not this |

For an optimizer method `converged` is true exactly when `gate_verdict` is
`pass`. A sampler row carries `gate_verdict = "n/a"`, and `converged` is its R̂
verdict (reported, and below the threshold).

`fit table` tabulates only completed method leaves whose method produces a
tabulated result. A leaf that is still running or incomplete, and a `pfilter`
leaf (a likelihood evaluation with no estimate to tabulate), get no row and no
message. A completed leaf whose result cannot be read is reported as a
`warning:` line and skipped; with `--quantity`, such a leaf is an error instead,
because dropping it would answer for fewer runs than were asked about.

To enumerate runs of any kind without the per-fit projection, use the generic
run browser: `camdl list --kind fit`. For one fit's full interpretation, use
`camdl fit summary <handle>`.

### 13.2 `camdl fit diff`

Compares two fit configs. Both must parse as complete v2 configs (`[estimate]`
is required), so this is a config-to-config diff, not a text diff.

```
$ camdl fit diff fit_a.toml fit_b.toml
diff: fit_a.toml → fit_b.toml

Model:
  /work/sir.camdl → /work/sir_waning.camdl (compiled model differs)
Data:
  cases: content differs
Parameters:
  beta: [estimate] → [fixed] = 0.31
  omega: [estimate] (added)
  sigma: [fixed] = 0.2 → [estimate]
  gamma: [fixed] = 0.3 (removed)
  N0: [fixed] 10000 → 20000

Method:
  chains: 1 → 4
```

The model is compared by its compiled identity, the same one a fit records, so
two paths to the same model read `(same compiled model)`; a `.camdl` source is
compiled with `camdlc` to get it, and a model that does not load is reported as
not compared rather than as unchanged. Data is compared by the content hash of
each stream the config binds. Parameter-side coverage: parameters entering or
leaving `[estimate]` and `[fixed]` (paired as a move when one does both),
changed fixed values, changed bounds (including the omit↔explicit transition,
which is meaningful because omitting means "fall back to the model file"), and
changed priors.

Method-side coverage is every key the `[method]` serializes to, one line per
changed key in key order, with nested tables flattened to dotted paths
(`gate.a_thresh`) and arrays diffed whole; a key one side lacks reads `null`. A
file with no `[method]` against one with reads `[method]: (new) pgas` /
`[method]: (removed)`:

```
$ camdl fit diff if2_a.toml if2_c.toml     # particles 1000→2000, cooling 0.70→0.95
diff: if2_a.toml → if2_c.toml

Model:
  (no model changes)
Data:
  (no data changes)
Parameters:
  (no parameter changes)

Method:
  cooling: 0.7 → 0.95
  particles: 1000 → 2000
```

Both halves are a rendering of the same typed diff `fit table --format json`
reports under `config_diff_from_baseline` (`ConfigDiff` in
`rust/crates/cli/src/fit/config_diff.rs`).

### 13.3 `camdl fit new`

Copies a fit config to a new path and injects a `[provenance]` block. It does
**not** change `[method]`, write a `starts` that names the source fit, or fill
in `reason`.

```
$ camdl fit new --from fit-if2.toml derived.toml
  [provenance] derived_from = "fit-if2.toml"
  hint: to warm-start the derived fit from a run of the source, write under its [method]
        starts = { from_posterior = "@<label>" }   # one draw per chain
        starts = { from_mle = "@<label>" }         # every chain at its estimate
        (a handle is @label, a run-id prefix or the run's folder; `camdl list` shows them)
created derived.toml
```

The written file is the source verbatim with:

```toml
[provenance]
derived_from = "fit_pgas.toml"
reason = ""
```

inserted before the first table. Refuses to overwrite an existing destination.
If the source already has a `[provenance]` block it says so and leaves it alone.

**The hint is the whole of the chaining story.** The derived file names its
source by handle under `[method]`, and the handle resolves when the derived file
runs (§6.6):

```toml
[method]
algorithm = "pgas"
…
starts = { from_posterior = "@<label>" }
```

or per run, as `camdl fit run … --starts from_posterior=@<label>`.

### 13.4 Reading results back

There is no `camdl summarize`. Four commands cover the ground:

| Command                     | Answers                                                                                                                 |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `camdl list`                | What runs exist? Filter by `--kind sim\|fit\|profile\|pfilter`, `--model`, `--scenario`, `--since 1h`. `--format json`. |
| `camdl show <TARGET>`       | Full metadata for one run, by short hash prefix or path.                                                                |
| `camdl cat <TARGET> [NAME]` | A stored file of the result; without a name, its summary file (a cell's trajectory TSV).                                |

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

`-b/--bundle` names the output (`-o` keeps its combined-table meaning on
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
`features`, `backends`, `data`, `dates`, `debugging`, `mre`, `model-comparison`.

---

## 14. Observation Semantics

An `observations { }` stream projects simulator state into the scalar
`projected` value that the likelihood scores. This section states, for each
projection kind, _what is read_, _when_, and _what resets afterwards_ — the
three facts that decide what a likelihood term actually means.

### 14.1 Projection kinds

There are five IR projection variants (`ir::observation::Projection`),
classified into two temporal kinds (`Projection::temporal_kind`). The
classification is **derived from the variant, never stored** — an
independently-stored kind could only ever disagree.

| DSL surface                                                                                                          | IR variant            | Temporal kind |
| -------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------- |
| `incidence(tr)` on an unstratified/one-cell transition, or `incidence(tr[a])`                                        | `cumulative_flow`     | Interval      |
| `sum(a in dim, incidence(tr[a]))`                                                                                    | `cumulative_flow_sum` | Interval      |
| `X[cell]`, or a bare `X` on an unstratified compartment                                                              | `current_pop`         | Instant       |
| `sum(X)` on a **stratified** compartment (a bare `X` there is E280), or a bare `X` pooling only `via` stages         | `current_pop_sum`     | Instant       |
| any expression over state — `X1 + X2`, `I / (S+I+R)`, `prevalence(of = I[a], among = N_local[a])`, a `let` reference | `derived_expr`        | Instant       |

Verified against the compiler on a two-stratum (`age = [child, adult]`) SIR:

```
projected = sum(I)                                 → current_pop_sum ['I_child', 'I_adult']
projected = I[child]                               → current_pop     I_child
projected = prevalence(of = I[child], among = N_local[child])
                                                   → derived_expr    pop I_child / binding_ref N_local_child
projected = sum(a in age, incidence(infection[a])) → cumulative_flow_sum ['infection_child', 'infection_adult']
projected = I[child] + I[adult]                    → derived_expr
```

- **Interval (incidence)** — the sum of per-transition flow counters accumulated
  over the period the row is declared to cover, read at the period's stop and
  **reset afterwards** (`StreamProjection::resets_after_observation`,
  `sim/src/inference/multi_stream_obs.rs`). The period is `[start, stop)` from
  the stream's `covers` form or its window columns; the accumulator also resets
  at the first period's start, so a warm-up before the first row is never
  scored.
- **Instant (prevalence / derived)** — a function of the state vector read at
  the observation instant. No accumulation, **no reset**; each observation is
  independent of the previous one.

Reset is per-stream and per-cadence: with several streams on different
schedules, only the Interval streams scheduled at the current union index are
zeroed. A **hole** in the data (an `NA` cell) contributes no likelihood term but
still resets, because the grid time is present.

**A bare `incidence()` over a stratified family is a hard error, not a silent
sum** (E280) — on an un-indexed stream, and on an indexed one, where it would
score every row against the pooled total and the hint names the stream's binder
(`incidence(infection[p])`):

```
$ camdlc check strat_probe.camdl        # projected = incidence(infection)
error[E280]: observation 'cases' is un-indexed, but `incidence(infection)` would silently
sum all 2 strata of 'infection' and apply reporting uniformly
```

Pooling across strata is a modelling decision, so the compiler makes you state
it — either `projected = sum(a in age, incidence(infection[a]))` with one
reporting rate in the likelihood, or an indexed stream `cases[a in age] { … }`
with `rho[a]`. The same rule covers state: a compartment written bare over a
population stratum (`projected = I` on an age-stratified `I`, or a `let` whose
body does so) is also E280, and `sum(I)` or `sum(a in age, I[a])` states the
pooling. Residence structure created by `via` may still be omitted — a bare `I`
on a staged, otherwise unstratified compartment pools its stages and compiles.
In model dynamics a bare stratified name is unchanged: it expands to `PopSum`.

**A stock as a count needs no operator; a stock as a proportion is
`prevalence(of = X, among = Y)`.** The proportion form is exactly `X / Y` — the
same resolution and the same `derived_expr` — plus three checks plain division
does not get: every cell `of` reads must be one `among` reads (E357; vacuous
when `among` reads no cell, such as a parameter or a table), both sides must
collapse the same dimensions (E358), and neither side may read a flow (E359).
The one-argument `prevalence(X)` and the multi-argument `prevalence(X1, X2)` are
removed (E356): they returned a count, exactly `X` or `X1 + X2`, under a name
that promised a proportion. A state expression over several compartments is
written as one — `projected = Y1 + Y2` — and the compiler-generated forms for a
`via erlang` (an `ESum` over the stage axis) or `via hyper_erlang` (an Add-chain
over branch-stage cells) compartment lower the same way.

`incidence` projects exactly one flow: several arguments is `E203`, and none is
`E250` (gh#669). A compartment population is an expression leaf, so state can be
added as an ordinary expression; a flow has no expression leaf naming it, so
flows are added term by term (`incidence(a) + incidence(b)`).

`current_pop` / `current_pop_sum` resolve to **integer** compartments
(`StreamProjection::from_ir`, `multi_stream_obs.rs`). A prevalence projection
naming a real-valued compartment is refused with a message saying so; use a
`derived_expr` (which reads both state kinds) instead.

Only `derived_expr` carries a `projection_state_grad`
(∂projection/∂compartment). The three selection variants are linear projections
and need none; the gradient-based samplers refuse a `derived_expr` whose
projection gradient is missing rather than scoring a silent-zero chain-rule
factor.

### 14.2 Snapshot timing

"State at `t`" means three different things depending on which path is running.
Getting this wrong changes what your data means, so it is stated per-path rather
than per-backend.

**Inference (particle filter, IF2, PMMH) — exact landing at `t`.** The filter
builds an `Exact`-policy timeline (`ExactInferenceTimeline::build`,
`intervention.rs`, called from `particle_filter.rs`) whose substep is
`dt.min(next_boundary − t)`, so the walk lands **exactly on the observation
time** even when it is off the `dt` grid. There is no snapping to a step
boundary. PGAS steps only on its grid, so the run is refused before it starts if
any period boundary lies off it (§6.5); a boundary it accepts is on a grid time,
so it too is scored at the time the row measures. Scoring happens after the
window walk completes, at the landed state. A scheduled effect whose fire time
is off the `dt` grid is refused loudly at timeline construction rather than
being silently re-tiled.

**ODE inference — the recorded output snapshot must land on the obs time.** The
deterministic likelihood (`compute_ode_loglik`, `inference/ode_loglik.rs`)
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
rounding (`x.max(0.0).round()` in `to_states`, `sim/src/ode.rs`). The gradient
path (`nl-lbfgs`) reads the **unrounded** continuous state instead
(`ode_loglik_and_grad`, `inference/multi_stream_obs.rs`). At small counts the
two therefore score slightly different likelihood surfaces for the same model
and data; at large `N` the difference is sub-nat.

**Forward `simulate --obs` — the recorded trajectory output grid.** Synthetic
observations are projected from the trajectory snapshots, not from the
integrator: `snap_at` (`cli/src/main.rs`) returns the **last snapshot at or
before** the observation time, and incidence is the difference of cumulative
flow between consecutive such snapshots.

**Consequence: every observation time must be a recorded output time, and a run
where one is not is refused (gh#589).** Reading the earlier snapshot would put a
flow's whole interval on one boundary and zeros elsewhere, and a stock would
step, while the file still carried the requested timestamps. So, with a weekly
output grid and a daily `emit_schedule`:

```camdl
output { trajectories { every = 7 'days } }
observations {
  prev { projected = I                      emit_schedule = every 1 'days  … }
  inc  { projected = incidence(infection)   covers = closing_at(time, 1 'days)   emit_schedule = every 1 'days  … }
}
```

`simulate` stops before simulating, with an error of this form (the first
off-grid time and its two neighbours are named):

```
error: observation stream 'inc' needs a value at t = 1, which is not a recorded output time (nearest recorded: 0 and 7).
  Required: every observation time must also be an output time.
```

`--output-every N` is subject to the same check, so a flag that reads as
presentational still constrains the observation cadence. **Set the trajectory
output cadence at least as fine as the finest `emit_schedule`** (or, when the
times come from a data file, fine enough to include them; `camdl predict`
records snapshots at the data times, §2.2.2). The output schedule defaults to
every 1.0 days, independent of `dt`, so sub-unit observation times need an
explicit `output { trajectories { every = … } }`.

### 14.3 Interaction with scheduled interventions

If a scheduled intervention fires at the same time as an observation, the
projection reads the **post-intervention** state. The data was generated in a
world where the intervention had already fired; scoring against the
pre-intervention state would deterministically bias the posterior against any
scenario that correctly represents it.

When each effect fires is decided once per run by the effect plan
(`rust/crates/sim/src/effect_plan.rs`, `EffectPlan::build`), on the backend's
clock: chain-binomial and every fit that steps it (`pfilter`, PMMH, IF2, PGAS)
fire on the run's grid `t_start + k·dt`, `k = ⌊(t − t_start)/dt + ½⌋`; the ODE
and Gillespie fire at the declared time (language spec §13.2). Effects at
`t_start` apply to the starting state before the `t_start` row is recorded or an
observation there is scored (§13.10).

The ordering at an instant is fixed in one shared dispatch seam
(`Schedule::arrive`), not by each backend: **effects before output**, each due
effect boundary's batch applied once, then all coincident outputs. On the
inference path the equivalent guarantee comes from the substep walk: a substep
that lands on one of the plan's firing instants fires that instant's batch
inside `process.step`, and scoring happens after the window walk. Forward
chain-binomial fires the plan's batch for each substep inside `step_one` and
records the output afterwards. (Historical: an earlier chain-binomial loop fired
interventions twice per scheduled time; see
`docs/dev/incidents/2026-04-17-chain-binomial-double-fire.md`.)

### 14.4 Likelihood families

Eight families (`ir::observation::Likelihood`): `poisson`, `neg_binomial`,
`normal`, `binomial`, `beta_binomial`, `beta`, `bernoulli`,
`zero_inflated_neg_binomial`.

Four facts about how they score that are not obvious from the names:

- **`normal(...)` is a discretized _count_ likelihood** (pomp / He et al.), not
  a continuous Normal PDF. It rounds the observation to the nearest non-negative
  integer and integrates the density over `[y − 0.5, y + 0.5]`. A clearly
  fractional observation logs a one-time warning (`eval_likelihood_resolved`,
  `sim/src/inference/obs_model.rs`). For a continuous outcome this is the wrong
  family.
- **Every count family rounds the observation.** `poisson`, `neg_binomial`,
  `binomial`, `beta_binomial` all apply `y.round().max(0.0)` before scoring (the
  log-pmfs in `obs_loglik.rs`, e.g. `negbin_logpmf`). A per-1000 rate or a
  scaled series fed to a count likelihood is silently rounded — `0.4 cases`
  scores as `0`.
- **`bernoulli(p = …)` thresholds at 0.5.** Any observed value `> 0.5` counts as
  a success (`eval_likelihood_resolved`, `obs_model.rs`), so a count column
  accidentally bound to a Bernoulli stream reads as "detected" for every
  non-trivial value.
- **`beta(mean = …, concentration = …)` requires `x ∈ (0, 1)` strictly**
  (`beta_logpdf`, `obs_loglik.rs`). An observed proportion of exactly 0 or 1 — a
  common outcome for small denominators — scores `−∞` and takes the whole fit
  with it. Use `beta_binomial` with the denominator as `n` when the data is `k`
  of `n`.

Pairing guidance:

- **Incidence:** `neg_binomial` or `poisson`, with the reporting rate in the
  mean (`mean = rho * projected`). Support on ℤ≥0; overdispersion natural.
- **Prevalence, single compartment:** `binomial(n = N, p = projected / N)` when
  the denominator is known and observed; `poisson` for large `N`. `neg_binomial`
  is valid but its dispersion means something different here than it does for
  incidence.
- **Prevalence as a fraction** (projection ∈ [0, 1]): `beta` for a directly
  observed proportion, `beta_binomial` for a `k`-of-`n` count.

**An incidence row whose period opens before the run is refused.** Every
incidence stream states what its rows cover (`covers`, or `window_start` /
`window_stop` columns — `docs/camdl-data-spec.md`), so each row's period is
known at load. A row whose period opens before `t_start` would be scored against
flow that was never simulated; the loader (`pfilter::stream_times_of_rows`)
refuses it before the filter runs, naming the period's start and `t_start`, and
the fix is to move `simulate.from` back or drop the rows the model cannot cover.
Under `closing_at(time, Δ)` a row labelled `t_start` is exactly this case. A
hole (`NA`) whose period opens before the run carries nothing to score and is
dropped instead. There is no zero-width bin to reason about: a row's period is
never inferred from its neighbours.

**The startup block prints the pairing — for two of the algorithms.** `pgas` and
`pmmh` print each stream's projection kind and likelihood family before running,
and warn when a `neg_binomial` is paired with a snapshot projection:

```
── method: pgas ──
  2 observation streams: detection, weekly_cases
  observations (2 streams):
    ✓ weekly_cases     incidence(infection)         NegBinomial
    ✓ detection        I                            Bernoulli
```

`camdl pfilter`, `if2`, `mh`, and the NLopt optimizers print only the
bound-stream line
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
├── simulate MODEL              forward simulation (single run, ensemble, sweep)
├── fit
│   ├── run CONFIG              run the [method] of a fit.toml
│   ├── summary FIT             R̂ / gate verdict / MLE table for one fit
│   ├── diff A B                compare two fit.toml configs
│   ├── table ROOT              one row per fit under a results tree
│   ├── new --from A DEST       derive a new fit.toml
│   ├── predict [FIT]           posterior-predictive (predicted vs observed)
│   └── methods                 list supported (algorithm, backend) pairs
├── pfilter MODEL               bootstrap particle filter at fixed θ
├── profile MODEL               profile likelihood over a parameter grid
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
  --dt DT                     step for discrete-time backends (default: the
                              model's dt, else 1.0; replaces the model's)
  --integrator rk4|rk45       ODE method override, refused off ode; tolerances
                              live in the model's
                              `simulate { integrator = rk45 { atol, rtol } }`
  --allow-degenerate-rates    restore legacy silent-zero on numerical collapse

Run window
  --to SPEC                   override the horizon (simulation.t_end). SPEC = a
                              model-time number, a date (date("YYYY-MM-DD") or bare
                              YYYY-MM-DD), or an observation anchor with an optional
                              offset ("last_obs + 8 weeks"). Anchored forms need
                              --fit for the observed times. A scenario declaring a
                              different horizon is an error
  --from SPEC                 override the start (simulation.t_start), the grammar
                              of --to ("first_obs - 1 week"; first_obs = the start
                              of the earliest observed period). Over the --fit
                              config's [simulate] from, over the model's own
                              `simulate { from }`. Keyed through the model.
                              Conflicts with --init-state
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
  --sweep PARAM=GRID          the DESIGN axis (repeatable): a comma list,
                              lin(min,max,n) or log10(min,max,n); several flags form
                              the name-sorted Cartesian product. Refused with --draws,
                              with a --param on a swept parameter, and with a scenario
                              that sets or scales one (§5.5)
  --sweep-file FILE           design points from a TSV: header = parameters, one row
                              per point. Refused with --sweep and --draws
  --parallel N                Rayon threads for the simulation phase; output is
                              byte-identical whatever the value (0 = all cores;
                              env CAMDL_PARALLEL; §4.4)

Trajectory output
  -o, --output FILE           export the combined table, rendered from the stored
                              cells (env CAMDL_OUTPUT)
  --stdout                    store the cell and print the combined table to
                              stdout. Single-cell only; conflicts with
                              --seeds/--replicates/--draws/--sweep, -o and --obs*
  --output-every N            one row every N time-units, overriding output { every }
  --no-flows                  drop every flow_* column
  --columns COL,...           restrict trajectory columns to this allow-list
  --dates                     add a calendar `date` column (requires model `origin`)

Observation output
  --obs FILE                  synthetic observations, all streams, one TSV
  --obs-dir DIR               one TSV per stream
  --obs-only FILE             like --obs, without the combined-table export
  --obs-only-dir DIR          like --obs-dir, without the combined-table export
  --design-from FIT_TOML      simulate on the observation design that fit config
                              binds — its observed labels, each row's own period,
                              its NA holes — instead of on `emit_schedule`.
                              Requires --obs-dir, --obs-only-dir or --projected-out;
                              θ from --params or --draws (one ds_NN/ per draw)
  --projected-out DIR         with --design-from: write each stream's projected
                              value (before observation noise) as one TSV per
                              stream, in column `projected`; stores nothing.
                              Usable with or without --obs-dir
  --emit-every N | NAME=N     override `emit_schedule`, in model time units, for
                              every stream or one by its observation-block label
                              (repeatable; the two forms are exclusive). Only a
                              recurring `every N` cadence can be overridden; a
                              stream declaring `at [...]` is refused by name.
                              Refused outright when the run emits nothing

Other artifacts
  --quantities-out DIR        emit quantities { } as <dir>/quantities/<name>.tsv +
                              quantities.json. A multi-scenario run writes one band per scenario
  --event-log [FILE]          store the lineage event log as a by-product of the cell;
                              pass a PATH to also export it. Single-run only
  --format parquet|tsv        event-log export format (default parquet); --tsv is
                              shorthand
  --reactive-log PATH         export the cell's reactive firing log (stored whenever
                              a reactive policy was active). Single-run only

Store
  --store DIR                 the run store's root, over a --fit file's `store`
                              key (§2.1)
  --force                     recompute every cell; the stored cell is moved to
                              quarantine first
  --label TEXT                name this invocation, found again as `@TEXT`; never
                              names a cell
  --manifest PATH             write the output manifest once the command succeeds
                              (§2.10); refused with --dry-run and --design-from
  --link-root DIR             link the stored result files under DIR (default
                              $CAMDL_LINK_ROOT; §2.11); refused with --dry-run and
                              --design-from
  --dry-run                   print the resolved plan and, for a multi-cell run,
                              which cells the store holds; writes nothing
```

`--draws posterior --fit` reads the fit from the run store. There is no
`--resume` flag: a run reads back the cells the store already holds by default,
and `--force` overrides it. `camdl fit run --sweep` sweeps a `[fixed]` parameter
of a fit (§7.3).

### `camdl fit run CONFIG`

```
  --seed N                    RNG seed (default 1)
  --parallel N                Rayon thread cap; bit-identical regardless of value
                              (default 0 = all cores; env CAMDL_PARALLEL)
  --force                     re-run; the stored run is quarantined
  --resume BASE_REF           extend a completed PGAS/PMMH run from a base run
                              (@label, run_id prefix or run folder); conflicts
                              with --force
  --sweep NAME=SPEC           Cartesian sweep over a fixed parameter (repeatable).
                              SPEC = V1,V2,... | lin(min,max,n) | log10(min,max,n)
  --label TEXT                name this invocation, found again as `@TEXT`
  --store DIR                 the run store's root, over the fit file's `store`
                              key (§2.1)
  --manifest PATH             write the output manifest once the command succeeds
                              (§2.10)
  --link-root DIR             link the stored result files under DIR (default
                              $CAMDL_LINK_ROOT; §2.11)
  --emit-every N | NAME=N     override a `[synthetic]` fit's observation emission
                              cadence (same grammar as `simulate --emit-every`);
                              refused on a fit with no `[synthetic]` block; the
                              fit re-keys

Chain starts
  --starts SPEC               uniform_unconstrained | lhs | uniform | from_prior | single
                              | from_posterior=<handle> | from_mle=<handle>
                              | from_params=<toml>. Default: from_prior when every
                              estimated parameter declares a prior, else
                              uniform_unconstrained
  --allow-nonconverged-source start from a from_mle / from_posterior source whose
                              stored verdict is not converged

Post-fit audit
  --no-dt-check               skip the Richardson dt-convergence check at θ̂
  --dt-check-strict           strict warning thresholds (0.5 nats chain_binomial,
                              0.1 nats ode) instead of the routine 2.0 / 0.5
  --dt-check-halvings N       default 2 (dt, dt/2, dt/4)

Per-algorithm overrides
  --decibans-thresh DB        gate's inter-chain loglik-spread floor
  --cooling-target-iters N    IF2 cooling target
  --max-tree-depth N          PGAS NUTS depth ceiling
  --trajectory-warmup N       CSMC-only sweeps before parameter updates
  --csmc-sweeps-per-nuts N    CSMC trajectory updates per parameter update
  --n-trajectories N          posterior trajectories saved
  --diagonal-mass             dense_mass = false (one-way)
  --no-nuts                   use_nuts = false — MH-within-Gibbs for θ|X (one-way)
  --no-ancestor-sampling      ancestor_sampling = false — plain particle Gibbs
                              (one-way; the run stores under its own address)
  --binomial btpe|btrs        chain-binomial sampler for pgas, pmmh, if2 and
                              pfilter (default btpe); not bit-compatible across
                              samplers, so it keys identity
  --no-adapt                  adapt = false — lock proposal SDs (one-way)
  --adapt-start N             step at which proposal-SD adaptation begins
  --record-ancestry           record ancestor indices for smoothing paths
  --record-prequential        record per-step predictive samples for `camdl compare`
```

### `camdl fit` — the other subcommands

```
camdl fit summary FIT
  FIT is a handle naming one run: @label, a run id prefix, or the run's folder;
  a fit.toml is refused with the runs it made listed
  --format text|json|md|latex (default text)
  --params-only               print only θ̂ as a flat params TOML (pipeable)
  --point posterior-mean|posterior-median
                              print a sampler run's posterior mean or median as a
                              params TOML with a [provenance] block (§2.2); with
                              --exclude-chains / --chains over the chains kept
  --allow-nonconverged-source with --point: summarise a run whose convergence
                              verdict is not converged (recorded in the block)
  --no-color                  disable ANSI colour (NO_COLOR is honoured regardless)
  --strict                    exit non-zero on provenance mismatch; auto-on when CI=true
  --exclude-chains IDS        recompute diagnostics over a chain subset (view only;
                              always warns; incompatible with --params-only)
  --chains IDS                the same over only the named chains; not with
                              --exclude-chains

camdl fit table ROOT          (see §13.1 for filters and formats)
camdl fit diff A B            (see §13.2)
camdl fit new --from A DEST   (see §13.3)
camdl fit methods             list (algorithm, backend) pairs with stability tiers

camdl predict RUN            RUN: the positional run handle, same grammar as `fit summary`
  --stream NAME               one logical stream, or an expanded leaf name
  --scenario NAMES            prospective scenario overlay (repeatable), applied on
                              top of the fit's own scenario/enable/disable (§11.3).
                              `fitted` is reserved for the no-overlay row: the
                              fitted model, the fit's own selection applied
  --enable / --disable NAME   ad-hoc overlay on the `fitted` row, on top of the
                              fit's selection; conflicts with --scenario
  --sweep PARAM=GRID          vary a parameter across the posterior (repeatable).
                              GRID = list | lin(min,max,n) | log10(min,max,n).
                              Free-forward, or forecast arms over a parameter read
                              only after the origin; same param in a scenario is an error
  --mode free_forward|one_step|forecast
  --horizon DURATION          the forecast's reach from the end of the training rows
                              (`4 weeks`, `28 days`); implies --mode forecast
  --design SOURCE=FILE        forecast: a design-reading stream's design values per
                              target, from FILE (prospective; repeatable)
  --realised-design           forecast: design values from the held-out rows (a
                              conditional score using realised future design)
  --covariate FORCING=STATEMENT
                              forecast: what a table forcing's values after the
                              origin are: prospective, realised or hold
                              (repeatable; required for each table forcing the
                              forecast reads after its origin; §2.2.2)
  --n-draws N                 posterior subsample cap for both horizons (default 200)
  --particles N               particles per draw in the one-step and forecast filters
                              (default: the PMMH fit's count, else 500)
  --seed N                    RNG seed for y_rep sampling (default 1)
  --exclude-chains IDS        drop chain ids (counted from 0) before banding; always warns
  --chains IDS                band over only the named chains; not with --exclude-chains
```

### `camdl pfilter MODEL`

```
--particles N               required
--params FILE / --param NAME=VALUE / --table NAME=FILE
--scenario NAME | --enable NAME | --disable NAME
--dt DT                     step for a model that declares none (default 1);
                            the model's declared dt is the step, and a
                            different --dt or --fit [config] dt is refused
--seed N (default 1)
--parallel N                Rayon threads, 0 = all cores (env CAMDL_PARALLEL)
--allow-degenerate-rates
--data [NAME=]PATH          observation TSV; repeatable. `--data PATH` binds the
                            single stream (or the one named by --obs); `--data
                            NAME=PATH` binds by name. Mixing the forms is an error
--obs NAME                  stream a bare `--data FILE` binds to
--fit PATH                  fit.toml supplying [data.observations] (used only when
                            no --data flag was given), the likelihood objective
                            (`ic_free`, always used) and [config]
                            allow_degenerate_rates (on if it or the flag is)
--time-format auto|numeric|date
--pf-max-substeps N         deterministic per-call compute budget
--binomial btpe|btrs        chain-binomial sampler (default: the --fit
                            method's, else btpe); btrs is faster, draws
                            different values, and keys identity
--replicates N              independent filter runs (default 1)
-o, --output FILE           default stdout
--trace FILE                per-observation diagnostics TSV ("-" for stdout)
--pf-health FILE            per-obs ESS + Snyder τ², with the implied
                            particles-to-avoid-collapse estimate exp(τ²/2)
--save-final-state FILE     final particle states; under the --fit config's
                            holdout_after, binds only the training rows and
                            saves the state at their end (gh#1138)
--save-paths FILE           smoothing-distribution trajectory samples
--n-paths N                 how many, for --save-paths (default 1)
--no-flows                  with --save-paths, write no flow_* column
--columns COL,...           with --save-paths, write only these compartment,
                            flow_<name> and noise_<name> columns
--save-filtering FILE       per-step particle states and log-weights
--save-filtered-streams FILE
                            per step and stream, quantiles of E[y_t | x_t, θ]
                            before (`predictive`) and after (`filtered`) y_t is
                            scored
--save-prequential STEM     {STEM}.tsv (joint score per step) + {STEM}.streams.tsv
                            (per-stream scores) + {STEM}.json
--no-save-samples           with --save-prequential, drop per-particle samples
--no-store                  compute in a temporary store, deleted at the end
--label L                   name this invocation, found again as `@L`
--store DIR                 the run store's root, over a --fit file's `store`
                            key (§2.1)
--manifest PATH             write the output manifest once the command succeeds
                            (§2.10); refused with --no-store
--link-root DIR             link the stored result files under DIR (default
                            $CAMDL_LINK_ROOT; §2.11); refused with --no-store
```

`camdl pfilter` stores its result in the run store (§2.1). The store's layout
and records are those of `docs/dev/proposals/2026-11-20-run-store-api.md`. A
filter run is one result, written in place, under
`<build>/pfilter/<model>/<settings>/params/seed_<seed>/`: its `loglik.toml`,
each replicate's log-likelihood in `replicates.tsv` under `--replicates N`, and
in `provenance.json` where each scored value came from and, under
`--allow-forcing-extrapolation`, each forcing the run held outside its range.
The compiled model and each data table are stored once each, under
`compiled_model/` and `data_table/`, and the run depends on them.

Every file the flags above write is a by-product of the run, stored inside its
folder under its kind (`trace/`, `pf_health/`, `final_state/`, `saved_paths/`,
`prequential/`, `filtering/`, `filtered_streams/`), and the flag exports a copy
to the path it names. Asking for a by-product never changes the run's id. A
rerun whose run and by-products are all stored computes nothing and writes the
exports from the stored copies; a rerun that asks for a by-product the stored
run lacks runs the filter again, checks that its log-likelihood reproduces the
stored one, and stores the by-product. `-o` and the terminal show the
log-likelihood rendered from the stored `loglik.toml` and `replicates.tsv`.
`camdl cat`, `show` and `list` find a filter run in the store `--store` names,
by id prefix, path or `@L`.

#### The saved paths file

`--save-paths FILE` writes `--n-paths` samples from the smoothing distribution
`p(x_{1:T} | y_{1:T}, θ)`, each traced back through the filter's ancestry. A row
is one path at one stop of the filter. The stops are the union of the
observation times of every bound stream, together with each declared period's
opening after the run's start, so binding another stream changes the rows. Two
comment lines open the file and name them. For an SEIR model with an
overdispersed infection, weekly incidence `cases` and a prevalence count every
10 days, the stops at 5 and 7 cut the first week into periods of 5 and 2 days:

```
# camdl-paths v2
# rows: the observation times of streams cases, prevalence; flow_* and noise_* cover [t_start, t_stop)
path	time	t_start	t_stop	S	E	I	R	flow_infection	flow_latency	flow_recovery	noise_infection	noise_dt_infection	cases	prevalence
1	5	0	5	974	12	7	7	16	4	7	0.7767931243130706	5	NaN	3.5
1	7	5	7	960	15	15	10	14	11	3	1.3567173763036673	2	5	NaN
```

The columns, in order:

1. `path`, from 1 to `--n-paths`. Each path is an equally weighted draw.
2. `time`, the stop, and `t_start`, `t_stop`: the half-open period
   `[t_start, t_stop)` that the row's `flow_*` and `noise_*` values cover.
   `t_stop` is `time`, `t_start` is the previous row's `time`, and the first
   row's `t_start` is the run's start.
3. Each integer compartment, the path's state at `time`.
4. `flow_<transition>` for every transition, in model declaration order: the
   number of times it fired during the row's period, the same quantity as
   `simulate`'s `flow_*` columns (§10.1). Flows add, so the flows of consecutive
   rows sum to the flow over their union, whatever the stops. The change in a
   compartment between two rows is the sum of its flows, plus whatever an event,
   intervention or `balance` moved; those are not transitions and are in no
   `flow_*` column.
5. `noise_<transition>` and `noise_dt_<transition>` for every overdispersed
   transition (`overdispersed(rate, σ²)`, language spec). On each substep of
   length `dt` on which the transition can fire, a unit-mean multiplier
   `g ~ Gamma(dt/σ², σ²/dt)` scales its rate. `noise_dt_` is the time within the
   period on which a multiplier was drawn, and `noise_` the multipliers'
   time-average there, `Σ g·dt / noise_dt_`. A substep on which the transition
   cannot fire draws nothing: its source compartment is empty or its rate is
   zero. `noise_` is then the realised environmental noise of the period,
   without the binomial or Poisson chance of the count. It is `NaN` when nothing
   was drawn (`noise_dt_` is 0), and exactly 1 when σ² = 0. Over several rows
   the noise is `Σ(noise_·noise_dt_) / Σ noise_dt_`, which is exact whatever the
   stops and whether or not the source emptied.
6. One column per observation stream: the stream's projection along the path at
   `time` (`E[y | x, θ]`), over that stream's own period, and `NaN` at a stop
   where the stream is not observed.

`--no-flows` drops every `flow_*` column and keeps the `noise_*` columns.
`--columns` keeps only the named compartment, `flow_<name>` and `noise_<name>`
columns; `noise_<name>` keeps its `noise_dt_<name>`, and the stream columns are
always written. A name the file cannot hold is refused with the list of the ones
it can. The filter records only the transitions the selection keeps, which
bounds the memory the paths take: one value per particle, stop and recorded
transition.

The saved paths are stored as the `saved_paths` by-product. Its identity holds
the filter run, `--n-paths`, `--no-flows` and the sorted `--columns`; none of
them changes the filter run's id. The invocation record keeps the stops' streams
and period openings as the by-product's labels,
`{"streams": [...], "period_openings": [...]}`, which `--manifest` exports
(§2.10).

`--no-store` computes in a temporary store and changes nothing else: the
log-likelihood is printed and every export is written. `camdl compare` passes it
to the filter runs it derives a trace from, so a comparison leaves no store
entries in the directory it was run from.

When the filter itself fails at the given parameters (its weights collapse, or
the state goes negative), `pfilter` exits 3; every other error exits 1.
`camdl compare`, which derives a prequential by running `pfilter` at each
posterior draw, reads that status: only a 3 is reported as a draw the filter
could not score, and any other failure is reported as itself.

With `--fit`, `pfilter` scores the likelihood that fit config defines: under the
config's own `scenario` / `enable` / `disable` (the interventions `fit run`
fired), and under `ic_free = true` it weights and resamples at `y₁` but leaves
`y₁`'s term out, so at the fit's MLE it reports the number `fit run` maximised
(inference spec §3.8). The config's selection and the `--scenario` / `--enable`
/ `--disable` flags combine as on `profile --fit`: either alone selects, the
same scenario named by both is one selection, and any other pair is refused — a
run scores one problem. A selection taken from the config applies its
intervention lists only (§11.3), so `--params` at the MLE is not overridden by a
fit scenario's `set`; `--scenario` beside `--params` stays refused (gh#194). It
prints the objective it scored, for example
`pfilter: objective: conditional on the first observation (ic_free = true in
fit.toml); omit --fit and pass --data for the full likelihood`,
and refuses an `ic_free` fit whose model has a deterministic `init { }`, as
`fit run` does for its `pfilter` method. Without `--fit` every observation is
scored. The stored leaf records the objective as `inputs.likelihood_objective`
(`full` or `conditional_on_first_observation`); the conditional objective is
part of the leaf's identity, and the full objective leaves the key unchanged.

### `camdl profile MODEL`

```
--sweep NAME=SPEC           required, repeatable (2D+ grids). SPEC = V1,V2,... |
                            lin(min,max,n) | log10(min,max,n). Every value must lie
                            in the parameter's bounds (the model's, narrowed by
                            --fit [estimate]); the swept name is recorded `focal`
--particles N               IF2 particles per cell: required for if2, refused for nl-*
--fixed NAME=VALUE          pin a parameter (repeatable); with --fit, a name the
                            config lists in [estimate] is refused (move it to
                            [fixed] in a copy of the config)
--fixed-file TOML           bulk form of the above (repeatable, later files win;
                            a fit artifact's [estimate] names are exempt)
--iterations N              IF2 iterations per grid point (default 50; refused for nl-*)
--n-starts N                independent starts per grid point (default 3)
--starts SPEC               where each cell's starts are drawn: the rule grammar
                            `fit run --starts` takes. Without it, the --fit
                            config's [method] starts; without either, the
                            default fit run takes: `from_prior` if every
                            parameter a cell estimates has a prior to draw
                            from, `uniform_unconstrained` otherwise. The run
                            says which. A source binds the cell's estimated set,
                            so a swept parameter needs no column in it
--starts-chains IDS         restrict a from_posterior rule to these chains of
                            its source, as on fit run (keyed; any other rule,
                            the default included, is refused)
--allow-nonconverged-source start from a from_mle / from_posterior source whose
                            verdict is not converged; without it the source is
                            refused, as on fit run
--cooling F                 IF2 cooling fraction (default 0.95), reached at
                            --cooling-target-iters as in a fit's `cooling`
                            (refused for nl-*)
--cooling-target-iters N    iteration the fraction is reached at (default 50;
                            refused for nl-*)
--rw-sd SPEC                NAME=SD,... or `auto`: every parameter that is not
                            swept, pinned (--fixed, --fixed-file, [fixed]) or a
                            model constant (`let`). An nl-* cell takes names
                            only (NAME=auto, or `auto`) and refuses an SD.
                            Optional with --fit: the estimated set is then the
                            config's [estimate] less swept and pinned names,
                            each IF2 step the --fit file's [method.perturbation]
                            entry (an if2 --fit) or the default; a --rw-sd SD
                            is natural-scale, converted to z at the base value;
                            a list given with --fit must name every such
                            [estimate] parameter. Required without --fit
--algorithm NAME            per-cell algorithm (default if2); pair with --backend
--backend NAME              chain_binomial (default) or ode
--fit PATH                  fit.toml supplying the estimated set, bounds, the
                            fixed list and [[simplex_groups]] (honoured by IF2;
                            every member must be estimated); an nl-* [method]
                            there also sets each nl-* cell's tolerance and
                            max_evals (else 1e-4 and 1500). Its holdout applies
                            to the bound data, --data included, as in fit run:
                            holdout_after removes every row ending after the cutoff
                            (keyed into the base), [data.holdout] files are
                            checked to lie after the training rows
--seeds SPEC                run the whole grid at each seed (multi-seed sensitivity)
--dt X                      step for a model that declares none (default 1.0);
                            every cell steps at the model's declared dt, and a
                            different --dt or --fit [config] dt is refused
--data / --obs / --time-format / --seed / --parallel / --scenario /
--enable / --disable / --table / --allow-degenerate-rates /
--pf-max-substeps / --binomial (if2 only; --binomial defaults to the --fit
method's sampler, else btpe)
-o, --output FILE           where the collected table goes (default stdout): one
                            row per grid cell, rendered from the stored points
                            (§2.8): swept values, best loglik over seeds and
                            starts, the estimates there, loglik_spread_starts,
                            starts_n_completed, n_starts, the best run_id
--format tsv|json           the table's format (default tsv)
--label TEXT
--manifest PATH             write the output manifest once the command succeeds
                            (§2.10)
--link-root DIR             link the stored result files under DIR (default
                            $CAMDL_LINK_ROOT; §2.11)
```

### `camdl compare [PATHS…]`

```
PATHS                       ≥2 when --config is absent. Each is a stored score,
                            prediction or fit run (its @label, a run-id prefix or
                            its folder), or a prequential.json (or a directory
                            holding one), read as it is
--mode one_step|free_forward|forecast
                            the mode whose scores are compared (default one-step
                            on the chain-binomial backend, free-forward on the
                            others; for a stored score or prediction, the first
                            of those it holds, else forecast)
--horizon DURATION          a forecast's reach past its origin; sets --mode
                            forecast, which needs it
--data STREAM=FILE          an outcome file per stream source (repeatable), read
                            by every compared prediction that binds the stream
--data-held-out STREAM=FILE a file of a stream's held-out rows alone (repeatable),
                            as `camdl score --data-held-out` reads it; one stream
                            by --data or by this, not both
--explain                   print the model-comparison methods guide and exit
                            (the same text as `camdl docs model-comparison`);
                            takes no PATHS
--config compare.toml       [[model]] entries; can also carry baseline/metrics/format/
                            streams
--baseline NAME             reference for Δ columns (default: argmax elpd)
--metric elpd,crps,wis,pit_cov90,cov50
--format table|md|json      (default table)
--streams NAME[,NAME...]    one table per observation stream (a leaf, or a source
                            for all its leaves), every row scored on that stream
                            alone; no figure combines streams
--allow-in-sample-ablation  under --streams, render rows whose fits assimilated
                            different streams although scored in-sample
--allow-mismatched-horizon  render even if T_score differs (Δ columns → '—');
                            waives only the times one row scored and the other
                            did not
--allow-data-mismatch       render even if the compared rows were bound to
                            different observed data, or scored different observed
                            values; the Δ is then confounded
--allow-mismatched-filter-settings
                            render even if the traces record different particle
                            or replicate counts
--draws N                   the posterior draws each sampler's fit run is
                            predicted from (`camdl predict --n-draws`; default
                            predict's); 1 is one posterior draw; an optimizer
                            run is predicted at its winner whatever N is
--replicates N              predictions per input at seeds seed..seed+N−1, each
                            scored against the first's outcome rows; log scores
                            combined by log-mean-exp, their totals give each
                            row's MC SE (default 5; 1 = one prediction, no MC
                            SE); an ODE free-forward score is predicted once
--allow-mixed-provenance    render rows whose provenance kinds differ
                            (plug-in vs posterior mixture) rather than refusing
--in-sample                 compare the training rows (role trained), stamped
                            in_sample, with the optimism caveat
--particles N               the filters' particle count of every prediction
                            compare makes (`camdl predict --particles`; default
                            1000); a stored prediction or score read in a
                            filtered mode at another count is refused before
                            anything is computed (stored inputs alone must
                            share one count); not under --mode free_forward
--seed N                    the seed of each fit run's prediction (default
                            predict's)
--allow-forcing-extrapolation
                            the `--allow-forcing-extrapolation` of each prediction
--exclude-chains [@FIT:]IDS per-fit (`@a:4`) or cohort-wide (`3,4`); mixing the two
                            forms is rejected. Always warns
--chains [@FIT:]IDS         keep only these chains, same forms; may name other fits
                            than --exclude-chains, never the same one
--pointwise PATH            write the per-observation Δelpd vector: joint scores
                            to PATH, per-stream scores to PATH with .streams
                            before its extension (under --streams: only the
                            .streams file, for the selected streams); refused for
                            a comparison of more than one table
--manifest PATH             write the output manifest of the predictions and
                            scores compared (§2.10); refused when they sit in
                            more than one run store
```

`compare` differences stored scores (§2.2.4). A stored score is read as it is. A
stored prediction is scored, and a stored fit run is first predicted, in the
mode `--mode` asks for, with `--particles` (default 1000) and `--seed` as the
`camdl predict` of that run would be given them, and `--draws` for a sampler's
run; an optimizer run is predicted at its winner whatever `--draws` is. Every
input is checked before anything is computed: a `--data` or `--data-held-out`
stream that no compared prediction binds is refused, naming it, and so is a
stored prediction or score read in a filtered mode whose filters ran at another
particle count than compare's (`--particles`, or its default when a fit run is
predicted; with neither, the stored inputs must share one count), naming each
input's count (unless `--allow-mismatched-filter-settings`). The prediction and
its score are computed in the process, by `predict` and `score`'s own
computations, and stored. Each is reused when stored, so a second `compare` of
the same runs computes nothing, and a prediction made with `camdl predict` and
the same options is the one `compare` finds.

A score's outcomes are, in order: the `--data` and `--data-held-out` files, each
given to every prediction that binds its stream (and left out of one that does
not), and a score of them that holds no row past the conditioning rows is
refused, naming `--in-sample`; when only the rows a prediction conditioned on
are compared (a fit that held nothing out, or `--in-sample`), those rows, which
need no file; otherwise the one stored score of the prediction that holds rows
past them. A stored fit keeps its held-out rows' dates and not their values
(§2.2.4), so a held-out comparison of a fit with no stored score and no outcome
file is refused, naming `--data`, `--data-held-out`, `camdl score` and
`--in-sample`; so is one whose prediction has several such scores, which names
them. A forecast's targets need outcomes in the same way.

The rows compared are a score's rows of the chosen mode, of one role: with
`--in-sample`, `trained`; otherwise `held_out` when the score has any, `unknown`
for a fixed point with no `--holdout-after`, and `trained` for the rest. A
forecast's rows are `future`, and each lead is its own table, in lead order; a
fit run's forecast's origin is the end of its training rows. Each row end is one
scored step: its log score is the score's `joint` row there, or the one stream's
own when one stream is scored there. A free-forward score holds no joint row, so
a joint table of a free-forward score of several streams observed at one time is
refused, naming `--streams`. Every number is the score's: no predictive sample
is read and no rule is recomputed. Every input is predicted `--replicates` times
(default 5), at the seeds `seed`, `seed+1`, …: a fit run from `--seed`, a stored
prediction or score from its prediction's seed, posed again with its other
options. Each prediction is stored and scored against the outcome rows of the
first score; each step's log score, and each stream's, is the log-mean-exp of
the replicates', the replicate totals give the row's filter-noise SE, and the
CRPS, WIS, PIT and coverage are the first replicate's. An ODE free-forward score
does not depend on the seed: it is predicted once, its `replicates` is 1 and it
has no SE. Under `--replicates 1` no filter-noise SE is measured: an evidence
cell of two filtered rows outside the se(Δ) band then ends
`filter noise not measured`, and the JSON row's `filter_noise_measured` is
`false`. `--draws 1` is one posterior draw, still provenance `posterior`. A
plug-in score at the posterior mean, or any one point, is a fixed point's: write
the point to a params file (for an optimizer run `fit summary --params-only`
prints its winner; it prints no point for a sampler run), predict it with
`camdl predict MODEL --params`, score it, and compare the score. Each note of a
score (a stream not scored, a draw whose filter degenerated) is printed on
stderr, prefixed by its row, and each forcing its computation held is a line
under the table on its row.

Two scores are differenced only when they agree on what they are of. A table
whose rows differ in any of these is refused (exit 2), the refusal naming the
field, and no flag waives it: the mode (`mode`), the role (`role`), a forecast's
lead (`lead`) and its reach as written (`reach`), whether a forecast read a
covariate stated `realised` (`covariates`), the predictive basis, and the
conditioning, so a stored fit's verified held-out rows (`hold_out_tail`) are not
differenced against a fixed point's declared ones (`hold_out_declared`). Inputs
holding different numbers of leads are refused, naming each input's leads. Rows
that scored different observed values of a stream at one time are refused
(`scored rows`) unless `--allow-data-mismatch` is given.

Columns: `role` and, for a forecast, `lead`, for rows read from stored scores;
`T_score`, `elpd`, `Δelpd`, `LR` (= exp(Δelpd), the likelihood ratio against the
baseline on the same scored observations — not an e-value: nothing here
establishes a null expectation ≤ 1, so it carries no anytime-valid reading),
`se(Δ)` (paired standard error over pointwise differences), `evidence` (decibans
plus a Jeffreys tier, or `within noise` when `|Δelpd| < 2·se(Δ)`), `crps`,
`Δcrps`, `wis`, `Δwis` (the weighted interval score), `PIT_cov90` (nominal 0.90;
below 0.70 triggers an overconfidence warning) and `cov50` (the share of
observations in the central 50% interval). `wis`, `Δwis` and `cov50` are shown
when a row records them, which a stored score does and a prequential.json does
not. The accuracy columns are per stream: they stay in the main table only when
every row scored exactly one stream, the same one; otherwise they move to a
block headed "Per-stream accuracy and calibration", one line per stream × model
with `n` (scored observations), `mean CRPS`, `Δ mean CRPS`, `mean WIS`,
`Δ mean WIS` (each Δ only when both rows scored the stream at the same times),
`PIT_cov90` and `cov50`, and the overconfidence warning names the stream. No
figure combines streams. A trace with no per-stream breakdown (schema 1) reports
CRPS and calibration as not available.

The `table` and `md` renderings carry the same footer, and it states only what
is specific to that comparison: the scored window and baseline, the conditioning
the rows were scored under (`in-sample`, or `held-out tail` with the sealed
`train_end`), the sort order, the optimism caveat, the miscalibration flags,
each trace's warnings, the filter-noise MC SE per row, the lag-1 autocorrelation
of the per-step Δelpd, and a closing pointer to `camdl compare --explain`. The
definitions those numbers rest on — the Jeffreys tiers and what they do and do
not calibrate, `se(Δ)`'s Newey–West/HLN construction and its small-`T`
unreliability (Sivula, Magnusson & Vehtari, arXiv:2008.10296) — live in that
guide (`docs/methods/model-comparison.md`), not under every table.

A comparison that renders only because an `--allow-*` flag waived a failed check
states each such override once in every format: one `⚠` line in the `table` and
`md` footers naming the flag, the rows it concerned, what differed and what that
does to the reading of the table, and one entry of the JSON's top-level
`overrides` array (`flag`, `rows`, `finding`, `consequence`; an empty array when
every check passed). A flag given to a comparison whose check passes waived
nothing and is not listed. The checks run once, before any rendering, and every
format states that one outcome.

The JSON keeps every field: `overrides`, `optimism_caveat`,
`evidence_scale_note`, `se_caveat` (populated below 100 scored steps),
`data_mismatch` and `data_unchecked`, `data_compared_on_prefix`,
`filter_settings_mismatch` and `filter_settings_unchecked`, and
`filter_noise_unassessed`, plus per-row `provenance`, `conditioning`,
`derivation`, `lag1_autocorr_delta_elpd`, `mc_se_elpd`, `warnings` and
`pit_cov90_warning`, a row read from a stored score adds `mode`, `role`, `lead`,
`reach` and `covariates_realised`, and per-row `per_stream` (one entry per
stream: `stream`, `n`, `mean_crps`, `crps_nonfinite`, `delta_mean_crps`,
`pit_cov90`, `pit_nonfinite`, `pit_cov90_warning`, and from a stored score
`mean_wis`, `delta_mean_wis` and `cov50`) with `per_stream_unavailable` naming
why it is empty. A comparison of more than one table (a forecast's leads) is
`{ "tables": [...] }`, each table the document above. The row-level `mean_crps`,
`delta_mean_crps`, `pit_cov90` and `pit_cov90_warning` are the single-stream
case's values and `null` otherwise. It is the machine surface and loses nothing
by repeating itself.

Two further refusals are overridable. A row read from a stored score records the
SHA-256 of the rows its prediction conditioned on, per source (a stored fit's
training rows, or a fixed point's data table), and rows whose digests differ are
refused unless `--allow-data-mismatch` is given; a prequential.json records none
and is named in `data_unchecked`. And two rows whose traces record different
particle or replicate counts are refused unless
`--allow-mismatched-filter-settings` is given, because the log of a
particle-filter likelihood estimate is biased downward by more at fewer
particles; a trace that records no settings (older than schema 5) is named in
`filter_settings_unchecked`.

Each step of a trace records its time `t`, the joint `log_score` (the log of the
joint one-step-ahead predictive density of every stream observed there), the
filter's `ess`, and `per_stream`: each scored stream's `y_obs`,
`y_pred_samples`, `log_score`, `crps`, `pit`, `interval` and `coverage`. A step
has no `y_obs`, `y_pred_samples`, `crps`, `pit` or `interval` of its own; a
trace written with those step-level fields still reads, and they are ignored.
The randomized-PIT uniforms are drawn one per scored stream value, step by step
and in stream order within a step, from `pit_randomization_seed`.

Each trace (schema 5) records how it was computed, as its `derivation` block:
`particles`, `replicates`, `dt`, `seed`, `theta` (`supplied`,
`posterior_mean { n_draws }`, `optimizer_winner` or
`posterior_draws { n_used, n_available }`), `excluded_chains`, `enabled` and
`disabled`, the replicate totals `replicate_elpd` and
`replicate_elpd_by_stream`, `mc_se_elpd` (the filter-noise SE, absent for one
pass), and `overrides` — the waivers applied to that one trace
(`in_sample_forced`, `forcing_extrapolation_allowed`), omitted when there are
none. A row read from a stored score fills it from the score and its prediction:
`particles` (0 for a free-forward score), `replicates`, `seed`, `theta`
(`posterior_draws`, or `supplied` for a fixed point), the chains and
interventions of the prediction, and `dt` null. A row's filter-noise SE is read
from this block, so an explicit `prequential.json` that carries one takes part
in the filter-noise gate.

Under `--streams`, each selected stream (leaf) gets its own table, and each
row's trace is re-scored on that leaf: a step that did not score it is dropped,
and a step that did keeps its `t` and `ess`, takes the leaf's own `log_score`,
and keeps only the leaf's `per_stream` entry. Nothing is recomputed. The
`derivation` block is restated for the leaf (`replicate_elpd` from
`replicate_elpd_by_stream` for that leaf, `mc_se_elpd` recomputed from it), an
`ess_collapse` warning is recounted over the surviving steps and
`starts_at_prior` kept only when the first step survives, and the trace records
`scored_streams = [leaf]`, so a re-scored trace is never paired with a joint
one. A name resolves to a leaf or to every leaf of a source, in the model's
order, and must resolve in every row; an explicit `prequential.json` accepts
only a leaf it scored. Every check runs per table, and a refusal names the
stream. The data check compares only that leaf's source, and a row with digests
but none for it is refused (`--allow-data-mismatch`). Rows whose fits
assimilated different leaves (the leaves their fit configs bind; unknown for an
explicit trace, which counts as different) are refused when in-sample unless
`--allow-in-sample-ablation` is given, whose override states the bias toward the
fit trained on fewer streams. Each table is headed by its stream; its footer
drops `t0` and, when the sets differ, lists each with its rows. The JSON is
`{ "streams_scored": [...], "tables":
[...] }`, each table the document above
plus `stream`, each row plus `streams_assimilated` (null when unknown); there is
no field combining streams. `--pointwise` writes only the `.streams` file.
Without `--streams` every output is unchanged, except that the stream-set
refusal names `--streams` with the streams the two rows share.

### Browsing and labelling

```
camdl list
  --store DIR                 the run store's root (default $CAMDL_STORE, else
                              ./camdl-store)
  --model SUBSTR   --scenario NAME   --since 1h|30m|2d
  --kind sim|fit|profile|pfilter|all                      (default all)
  --limit N (default 50) | --all
  --format human|json

camdl show TARGET             --store DIR   --format human|json
camdl cat HANDLE [NAME]       --store DIR   --kind KIND   --run RUN_ID
camdl cat --manifest M NAME   --store DIR   --kind KIND   --run RUN_ID   --each
  --chains IDS | --exclude-chains IDS        rows of a fit run's draws
camdl path HANDLE NAME...     --store DIR   --kind KIND   --run RUN_ID   --all | --each
camdl path --manifest M NAME...
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
camdl dev eval MODEL --expr "a,b"
  --params FILE / --param NAME=VALUE / --table NAME=FILE
  --from T (0) --to T (100) --every DT (1)   |   --at T1,T2,…  (mutually exclusive)
  -o FILE                     default stdout
camdl dev compile / camdl dev doctest      (see "Compiler passthroughs")
```

### Sensitivity analysis

Sensitivity analysis (Sobol indices and similar) is not a camdl concern, and
camdl generates no space-filling designs. Generate the design in R or Python,
run it with `camdl simulate --sweep-file design.tsv -o design_out.tsv`, and
compute the indices over the combined table (every row carries its `point_id`
and `sweep:<param>` values) with R's `sensitivity` package or Python's `SALib`.

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
`camdl profile --fixed-file`, `camdl fit run --starts from_params=<file>`, and
fit.toml's `[fixed] from_file`. `--params` is repeatable; later files override
earlier ones.

**Values must be numbers.** A unit-annotated string is a hard error, not a
silent parse:

```
$ camdl simulate model.camdl --params bad.toml     # gamma = "0.1 /day"
error: schema mismatch in bad.toml: bad.toml:gamma: expected a number or table section, got String("0.1 /day")
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
resolution chain — and the same refusal (§1.3, gh#571): a row naming a parameter
a selected scenario `set`s or `scale`s is refused before anything runs, naming
the file and the row, since the row would replace the scenario's value in every
cell. On every other parameter the row wins over the lower tiers. The
`--param-vec` unknown-name check runs during full parameter resolution, which
`--dry-run` short-circuits: a `--dry-run` invocation will not report a bad key.
