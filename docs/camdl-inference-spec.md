# camdl Inference Workflow Specification

**Version:** 0.1-draft **Date:** 2026-03-30 **Depends on:** Experiment Spec v0.5
(output schemas), VOI Spec v0.1

---

## 1. Design Principles

**Files are the workflow.** Every inference step reads files and writes files.
The pipeline exists in the filesystem, not in the researcher's head. Any step
can be rerun. Any result can be traced to its inputs.

**params.toml is the universal interface.** The inference pipeline's final
output is a params.toml. It feeds directly into `camdl simulate`,
`camdl batch run`, `camdl voi run`. No format conversion. One chain from raw
data to decision.

**The fixed/free partition is explicit and exhaustive.** Every model parameter
must be declared as either estimated or fixed. No silent defaults. Accidental
fixation — the most common calibration error — is a compile-time error, not a
runtime surprise.

**Provenance is structural.** Every machine-generated file includes a content
hash and input references. Modified files are detectable. The full chain from
data to decision is auditable.

**TOML for humans, JSON for agents, TSV for data.** No file serves two masters.

---

## 2. File Roles

| File                   | Format | Written by | Read by                 | Purpose                      |
| ---------------------- | ------ | ---------- | ----------------------- | ---------------------------- |
| `fit.toml`             | TOML   | Human      | All fit commands        | Inference configuration      |
| `fit_state.toml`       | TOML   | Each stage | Next stage              | Inter-stage handoff          |
| `{stage}_summary.json` | JSON   | Each stage | Agent / status          | Machine-readable diagnostics |
| `mle_params.toml`      | TOML   | validate   | experiment system       | Final MLE (IS a params.toml) |
| `fit_record.json`      | JSON   | validate   | Audit / archive         | Self-contained provenance    |
| `fit_report.txt`       | Text   | validate   | Human / methods section | Human-readable summary       |
| `parameter_traces.tsv` | TSV    | Each stage | Plotting                | Per-iteration param values   |
| `diagnostics.tsv`      | TSV    | Each stage | Plotting / analysis     | Per-iteration ESS, loglik    |
| `profiles/*.tsv`       | TSV    | validate   | Plotting / CI           | Profile likelihood curves    |

---

## 3. The Fit File

### 3.1 Structure

```toml
[fit]
model = "models/he2010_london.camdl"
output_dir = "fit/he2010"

# Data sources — one per observation block in the model.
# Keys must match observation block names in the .camdl file.
[data]
weekly_cases = "data/london_cases.tsv"

# For multi-stream models (polio):
# [data]
# afp_cases = "data/afp_by_district.tsv"
# es_positive = "data/es_results.tsv"

# Optional: out-of-sample holdout data.
# Keys must match [data] keys. Scout/refine never see holdout data.
# Validate runs PF on train + holdout and reports separate logliks.
# [holdout]
# weekly_cases = "data/london_cases_holdout.tsv"

[config]
dt = 1.0
# The forward backend for [synthetic] data generation is [synthetic].backend
# (gh#241); fit stages declare their own `backend`.

# ── Estimated parameters ──────────────────────────────
# Every parameter here will be perturbed during IF2.
# rw_sd is optional: if omitted, auto-computed from parameter bounds
# as (hi-lo)/6 on the transformed scale (tier 1 heuristic).
# If specified, rw_sd is on the NATURAL scale (converted internally
# via delta method).
[estimate]
R0 = { rw_sd = 5.0 }
sigma = {} # auto rw_sd from bounds
gamma = {} # auto rw_sd from bounds
rho = { rw_sd = 0.02 } # explicit override
amplitude = {}
S0 = { ivp = true } # auto rw_sd, perturbed only at t=0
E0 = { ivp = true }
I0 = { ivp = true }

# Optional: override starting value (default: from model)
# sigma = { start = 0.1 }

# Optional: override transform (default: derived from parameter type)
# R0 = { rw_sd = 5.0, transform = "identity" }

# Optional: narrow the search region (default: full model bounds)
# R0 = { rw_sd = 5.0, bounds = [30.0, 80.0] }
# gamma = { bounds = [0.05, 0.15] }

# ── Fixed parameters ──────────────────────────────────
# Every parameter here is held constant at its model default.
# The user MUST explicitly assign every model parameter to
# either [estimate] or [fixed]. Missing parameters are an error.
[fixed]
N0 = true
mu = true
k = true
psi = true
cohort = true
```

### 3.2 Search Bounds

The model declares structural bounds: `R0 : positive in [1.0, 100.0]` — the
physically valid range.

The fit.toml can declare narrower **search bounds** — "for this data, focus the
search here":

```toml
R0 = { rw_sd = 5.0, bounds = [30.0, 80.0] }
```

Search bounds are optional on every parameter. When omitted, the full model
bounds apply. Typically only 1-2 parameters need narrowing — those where prior
knowledge constrains the plausible region.

**Rules:**

1. **Fit bounds must be within model bounds.** `bounds = [0.5, 80]` on a
   parameter declared `in [1.0, 100.0]` is an error:

   ```
   error: fit bound [0.5, 80] for 'R0' extends below model bound [1.0, 100.0].
     Fit search bounds must be within model structural bounds.
   ```

2. **Fit bounds control the transform.** When `bounds` is specified, the
   logit/scaled-logit transform maps to that interval instead of the model
   bounds. This concentrates the random walk in the region of interest rather
   than wasting perturbation on implausible values.

3. **Fit bounds control scout random starts.** Scout initializes chains from
   uniform random within the fit bounds, not the full model bounds.

4. **Boundary proximity warning.** If the MLE is within 1% of a fit bound, warn
   that the true MLE may be outside the search region:

   ```
   ⚠ R0 = 30.2 is within 1% of fit lower bound (30.0).
     The MLE may be outside the search region.
     Consider widening: R0 = { bounds = [15.0, 80.0] }
   ```

   This is reported in `{stage}_summary.json` and `camdl fit summary`:

   ```
   R0 = 30.2   rw_sd=5.0  bounds=[30, 80]  ⚠ AT LOWER BOUND
   ```

   The warning is the safety net. Without it, narrowed bounds can hide the true
   MLE. With it, they're a useful search tool.

### 3.3 Exhaustive Partition Rule

On `camdl fit scout` (or any fit command), the tool checks:

```
model_params = set of all parameters declared in .camdl
estimated    = set of keys in [estimate]
fixed        = set of keys in [fixed]

if estimated ∪ fixed ≠ model_params:
    missing = model_params - estimated - fixed
    error:
      "Parameters not assigned in fit.toml: {missing}
       Every model parameter must appear in [estimate] or [fixed].
       
       Add to [estimate]:  {p} = {{}}
       Add to [fixed]:     {p} = true"
       
if estimated ∩ fixed ≠ ∅:
    overlap = estimated ∩ fixed
    error:
      "Parameters in both [estimate] and [fixed]: {overlap}
       Each parameter must appear in exactly one section."
```

This is a hard error. No warnings, no defaults. The user must explicitly decide
the fate of every parameter.

### 3.4 No `params` Field

Starting values come from the model's parameter defaults (declared in the
`.camdl` file or in params files referenced by the model's `simulate {}` block).
The fit.toml does not reference params files.

If the user wants a non-default starting value, they use `start` in the
`[estimate]` section:

```toml
sigma = { rw_sd = 0.005, start = 0.1 }
```

This keeps the fit.toml self-contained: it says what to estimate and how, not
what the values are.

### 3.5 Observation Model

The observation model is declared in the `.camdl` file's `observations {}`
block. The fit.toml references it by name:

```toml
[data]
weekly_cases = "data/london_cases.tsv"
```

The key `weekly_cases` must match an observation block name in the model. The
data file provides (time, value) pairs. The observation model (likelihood
family, projection, variance formula) comes from the DSL — it's part of the
model specification, not the fit configuration.

For models with multiple observation streams:

```toml
[data]
afp_cases = "data/afp_by_district.tsv"
es_positive = "data/es_results.tsv"
```

The particle filter sums log-likelihoods across all streams at each observation
time.

### 3.7 Replicate fits and synthetic-data calibration

A fit config can describe a grid of fits instead of a single fit, varying two
orthogonal axes:

- **Data axis** — real data (`[data]`) or synthetic data (`[synthetic]`,
  generated from known truth). Mutually exclusive.
- **Fit axis** — list-valued `fit_seeds`. Each seed runs the full stage pipeline
  independently with a different IF2/PGAS seed and (optionally) perturbation
  start.

The grid is the Cartesian product. Collapses orthogonally: omit both and the fit
runs once, unchanged from today.

#### Config shape

```toml
# Top-level scalars must precede any [table] header.
fit_seeds = [101, 102, 103] # optional; list-only

[model]
camdl = "models/sir.camdl"

# Choose one of [data] or [synthetic], never both.
[data.observations]
cases = "data/cases.tsv"

# …or:
[synthetic]
true_params = "params/truth.toml"
sim_seeds = "1:20" # or an explicit list
datasets = 20 # optional; inferred from sim_seeds
scenario = "baseline" # optional
```

#### Canonical modes

| Mode                    | `[synthetic]` | `fit_seeds`     | Output layout                     | Cells |
| ----------------------- | ------------- | --------------- | --------------------------------- | ----- |
| Single fit              | —             | scalar / absent | `real/fit_<seed>/`                | 1     |
| Start-sensitivity       | —             | list, len M     | `real/fit_<seed>/` × M            | M     |
| SBC (classical)         | N datasets    | scalar / absent | `synthetic/ds_NN/fit_<seed>/`     | N     |
| SBC × start-sensitivity | N datasets    | list, len M     | `synthetic/ds_NN/fit_<seed>/` × M | N × M |

#### Output directory

Every fit lives under a data-source subdirectory. No single-fit exception — the
seed is always the leaf:

```
results/fits/<name>/
  real/                       or    synthetic/
    fit_101/                            ds_01/
      scout/ refine/ …                    fit_101/
    fit_102/                                scout/ refine/ …
      …                                   fit_102/ …
    summary.tsv                         …
                                        summary.tsv
                                        coverage.tsv
                                        truth.toml
                                        data/
                                          ds_01.tsv
                                          …
```

`summary.tsv` (always): one row per `(dataset, fit_seed)` with every estimated
parameter's MLE, the log-likelihood, and the content hash.

`coverage.tsv` (synthetic mode only): one row per estimated parameter with
`truth`, `mean_mle`, `bias`, `sd_mle`, `q05`, `q95`, `covers_truth`,
`n_datasets`. `covers_truth = 1` when the central 90 % MLE window brackets the
declared ground truth.

#### Synthetic data semantics

When `[synthetic]` is present, the runner generates `len(sim_seeds)` datasets
before dispatching any fits. Each dataset is a single run of the chosen backend
at `true_params` with `sim_seed`, sampled through every declared observation
stream at its declared schedule. The resulting wide-format TSV is written to
`<fit_dir>/synthetic/data/ds_NN.tsv` and handed to the fit runner as if the user
had supplied it via `[data.observations]`.

Generation is deterministic: same `(true_params, sim_seed)` produces
bit-identical data. The content hash of each dataset participates in the
per-cell provenance hash so regenerating with unchanged inputs is a cache hit,
not a redo.

#### Validation

- `[data]` + `[synthetic]` — hard error (choose one).
- Neither present — hard error.
- `sim_seeds` range/list with duplicates — hard error (would collide on
  provenance hashes).
- `datasets` supplied and `≠ len(sim_seeds)` — hard error.
- `fit_seeds` list with duplicates — hard error.

### 3.8 IC-free inference

Stochastic compartmental models have a ridge in `(β, I₀)`: short observation
windows admit many parameter combinations that fit the data equally well. Fixing
`I₀` gives a tight `β` posterior that is paid for with a pretence; estimating
`I₀` reveals the ridge but needs a prior on `I₀` to resolve along it. The
principled alternative from the pomp literature (King et al. 2008; Bretó et
al. 2009) is to condition the likelihood on the first observation and let the
particle filter's initial reweight pin `I₀` implicitly:

```toml
ic_free = true # top-level key — fit.toml is parsed strictly (deny_unknown_fields); there is no [fit] table
```

When set:

- The PF / IF2 / PGAS runs still weight and resample at `y₁` (that's what pins
  `x₀` given `y₁` via Bayesian update on the particle cloud).
- The log-likelihood accumulation skips `y₁`. The returned loglik is
  `log L_c(θ | y₁) = Σ_{t=2}^T log p(y_t | y_{1:t-1})` rather than the full
  `log L(θ)`. These differ by `log p(y_1)` and are not directly comparable
  across runs with different `ic_free` settings.

Precondition: at least one `[estimate.*]` entry must be `ivp = true`. Without
per-particle spread at `t=0`, the first reweight cannot discriminate between
particles and `ic_free` silently degenerates to dropping the first observation.
Config validation rejects the degenerate case:

```toml
[estimate]
I0 = { bounds = [1, 500], ivp = true } # required under ic_free
```

See `docs/dev/proposals/2026-04-18-ic-free-inference.md` for the mathematical
derivation, references (King Nguyen Ionides 2016 JSS, Ionides et al. 2015 PNAS,
King et al. 2008 Nature, Bretó He Ionides King 2009 AOAS), and the
epistemic-laundering motivation.

Explicitly not the same: the Cori-Fraser-Cauchemez / EpiEstim renewal-equation
approach avoids `I₀` by replacing the compartmental structure during growth with
a branching process parameterised by `R_t`. That is a different process model,
not a different likelihood factorisation, and is out of scope for this feature.

### 3.9 The conditioning boundary (covariate-informed burn-in)

`simulate.from` (the model origin `t_start`) and the first observation need not
coincide: a model may start dynamics well before the data — e.g. to let births
and SIA/MCV covariates shape the susceptible pool over a span where no case data
exists yet. The particle filter conditions observation _k_ over the half-open
window `(t_{k-1}, t_k]`, so the first window is `(t_start, first_obs]`. For an
**incidence** observation (a flow accumulated between observations), that first
window then spans the whole pre-data gap, and the first datum is scored against
the flow integrated over the entire gap — a wrong likelihood (gh#134).

`condition_from` decouples where dynamics begin from where the likelihood
begins. It places a **conditioning boundary** one cadence before the first
datum: the leading span `[t_start, condition_from)` becomes a **warm-up** —
simulated with the full stochastic dynamics (births, campaigns, seasonality,
process noise) but not scored — and the first observation is scored against one
normal cadence `(condition_from, first_obs]`.

```toml
condition_from = "first_obs - 1 week" # the default for every stream
# condition_from = "date(\"2014-08-18\")"   # or an absolute calendar date
# condition_from = "19"                      # or an absolute model-time number (quoted)
```

For a multi-cadence model the conditioning window is **per observation stream**.
A table form carries an optional all-streams `default` plus per-stream
**shadows** keyed by the observation-block label (its `[data.observations]`
key); a stream with no shadow falls to the `default`:

```toml
[condition_from]
default = "first_obs - 1 week" # applied to every stream …
es = "first_obs - 2 weeks" # … except `es`, which this shadows
```

Conditioning is **explicit, not inferred**: an incidence stream whose first
observation lands anomalously far after `t_start` (a wide leading window
relative to its own cadence) with _no_ conditioning window is a **hard error**
(W329) naming the `condition_from.<label> = …` fix — the single first datum
cannot constrain that whole warm-up span, and an _inferred_ boundary would fail
silently on irregular data. A stream whose first observation is ~one cadence
after `t_start` needs nothing.

Mechanically the boundary is a leading **reset-only hole** on the shared
observation grid: its grid time resets the incidence accumulator (so the first
scored bin is one cadence) while contributing no likelihood term (the warm-up
flow is discarded, not scored). It rides the same hole/reset seam as sparse
(`NA`) observations, so every algorithm — bootstrap PF, IF2, correlated-PF,
PGAS, PMMH — receives it through the bound observation set.

- **Domain.** `condition_from` must resolve strictly between `simulate.from` and
  the first observation, and onto the `dt` grid. A value at/after the first
  observation, at/before the origin, or off-grid is a hard error naming the
  valid range.
- **Default.** Omitted (or resolving to `t_start`) inserts no hole — byte-for-
  byte identical to no conditioning.
- **Guard (W329).** When `simulate.from` sits a wide gap before the data on an
  incidence stream and `condition_from` is unset, the fit is rejected with a
  hard error naming this fix; for a prevalence observation (read at the instant,
  no accumulation) it is only a soft warning.
- **Composition with `ic_free`.** The two cannot be combined: the leading hole
  is obs-index 0, and `ic_free` conditions on the first observation there — a
  hole is not an observation, so the fit is rejected ("nothing to condition
  on"). They address overlapping uncertainty anyway: a covariate-informed
  burn-in _derives_ the boundary state, reducing the need for the `ic_free`
  initial-state estimate.

See `docs/dev/proposals/2026-06-09-burnin-conditioning-window.md` for the
design, the gh#134 reproduction, and the per-algorithm mechanics.

---

## 4. The Fit State File

### 4.1 Purpose

`fit_state.toml` is the inter-stage handoff. Each stage reads the previous
stage's fit_state.toml and produces its own. The user never needs to edit it
(but can, for debugging).

### 4.2 Schema

```toml
# <fit_dir>/real/fit_<seed>/scout/fit_state.toml
# Auto-generated by camdl fit scout.

stage = "scout"
seed = 1
timestamp = "2026-03-30T14:30:00Z"
best_loglik = -3891.2
initial_loglik = -4523.7
best_chain = 4
n_chains = 8
n_good_chains = 6

[start_values]
# Best chain's best-loglik parameters (not endpoint — the point
# during the chain's trajectory where loglik was highest)
R0 = 55.3
sigma = 0.081
gamma = 0.084
rho = 0.49
amplitude = 0.52
S0 = 71200
E0 = 140
I0 = 118

[rw_sd]
# rw_sd from the user's fit.toml (or auto from bounds if omitted).
# No inter-stage auto-calibration — each stage uses the user's
# specified values. Set explicit rw_sd after inspecting scout traces.
R0 = 8.5
sigma = 0.003
gamma = 0.004
rho = 0.025
amplitude = 0.02
S0 = 3200
E0 = 95
I0 = 45

[tail_chain_agreement]
# Per-parameter Â over the last half of iterations. Refine reads
# this to gate Stage 1 of its compound scout-convergence check.
R0 = 1.02
sigma = 1.01
gamma = 1.07

# Per-chain CLEAN-EVAL log-likelihoods + standard errors (in
# chain-id order), produced by the Step-7 loglik-eval re-scoring.
# The compound scout-convergence gate combines these with
# tail_chain_agreement to compute an SE-aware decibans-spread
# threshold (see §6.1.1).
chain_eval_logliks = [
  -3893.4,
  -3891.2,
  -3897.8,
  -3895.1,
  -3892.0,
  -3899.4,
  -3895.7,
  -3894.8,
]
chain_eval_ses = [0.5, 0.4, 0.6, 0.5, 0.4, 0.7, 0.5, 0.5]

ivp_params = ["S0", "E0", "I0"]

# Resolved compound-gate config (Phase 3): the values actually in
# force at runtime, after the priority chain
#   CLI flag > [stages.<stage>.gate] > GateConfig::default()
# collapsed. Persisted so `camdl fit summary` renders the verdict
# against the threshold the run was actually judged by, not whatever
# fit.toml says at summary-time. Absent on legacy fit_state.toml
# files; summary then falls back to GateConfig::default() with a
# "(thresholds unknown)" caveat.
[resolved_gate]
a_thresh = 1.01
decibans_thresh = 30.0

[resolved_loglik_eval]
n_particles = 4000
n_replicates = 8
combine = "log_mean_exp"
```

### 4.3 How Stages Chain

```
fit.toml
  │
  ├─ camdl fit run fit.toml --stage scout
  │    reads: fit.toml (config + rw_sd)
  │    writes: <fit_dir>/real/fit_<seed>/scout/fit_state.toml
  │
  ├─ camdl fit run fit.toml --stage refine
  │    [stages.refine] init = "from_mle", init_mle = "scout"
  │    reads: fit.toml (config) + scout/fit_state.toml (start_values, rw_sd)
  │    writes: <fit_dir>/real/fit_<seed>/refine/{fit_state.toml, mle_params.toml}
  │
  └─ camdl fit run fit.toml --stage validate
       [stages.validate] init = "from_mle", init_mle = "refine"
       reads: fit.toml (config) + refine/fit_state.toml (start_values, rw_sd)
       writes: <fit_dir>/real/fit_<seed>/validate/{fit_state.toml, mle_params.toml,
               profiles/, fit_record.json}
```

> **v2 layout note.** Stage directories live under
> `<fit_dir>/real/fit_<seed>/<stage>/` (or
> `<fit_dir>/synthetic/ds_NN/fit_<seed>/<stage>/` for SBC replicates). The
> `real/fit_<seed>/` wrapper was introduced in commit `5f1e704` (2026-04-18);
> diagrams that show stages directly under `<fit_dir>/` are pre-v2 and no fit
> produced by `cmd_fit_run_v2` writes that shape.

When the stage declares `init = "from_mle"` +
`init_mle = "<prior-stage-or-dir>"`:

- Starting values come from the prior stage's `fit_state.toml [start_values]`
  (overrides model defaults and fit.toml `start` fields)
- rw_sd comes from that `fit_state.toml [rw_sd]` (overrides fit.toml rw_sd
  values)
- Explicit `--rw-sd` CLI flags override everything

When no `init_mle` is declared (default `init = "uniform_unconstrained"`, etc.):

- Starting values come from model defaults + fit.toml `start` fields
- rw_sd comes from fit.toml `[estimate]` rw_sd values

---

## 5. MLE Output and Provenance

### 5.1 `mle_params.toml`

The final output of inference. A valid params.toml with provenance in comments:

```toml
# camdl fit output
# Content hash: a3c1e890 (editing any value below invalidates this)
# Input hash: 7f2c1d3a (identifies the computation that produced this)
# Model: models/he2010_london.camdl (hash: 3a7f2c1d)
# Data: data/london_cases.tsv (hash: f9e2b047)
# Fit: fit.toml (hash: b2c4d6e8)
# Seed: 1
# Stage: validate, chain 2
# Log-likelihood: -3804.9 (sd: 5.2, N=10000)
# ESS at MLE: mean=3842, min=1205
# Timestamp: 2026-03-30T14:30:00Z
# camdl version: 0.3.0

R0 = 56.82
sigma = 0.0791
gamma = 0.0832
rho = 0.488
amplitude = 0.554
S0 = 73151
E0 = 127
I0 = 127
N0 = 2462500
mu = 0.0000548
k = 50.0
psi = 0.116
cohort = 0.0
```

### 5.2 Two Hashes for Two Purposes

**Input hash (in `fit_record.json`):** identifies the computation.

```
input_hash = sha256(
    model_ir_bytes +
    sorted(data_file_bytes for each stream) +
    fit_toml_bytes +
    seed +
    camdl_version
)[:8]
```

Answers: "what computation produced this?" Includes the seed so that two runs
with different seeds have different input hashes. When no `--seed` is given, one
is generated, recorded, and included.

**Content hash (in `mle_params.toml` header):** detects manual edits.

```
content_hash = sha256(
    sorted(param_name + "=" + param_value for each parameter)
)[:8]
```

Answers: "have the output values been changed since the fit wrote them?"
Computed purely from the parameter values in the file.

If the user edits any parameter value, `camdl fit status` detects the mismatch:

```
$ camdl fit status fit.toml

  validate/mle_params.toml: ⚠ MODIFIED
    R0: 56.82 → 60.0 (manual edit)
    Content hash mismatch: expected a3c1e890, computed d4f7b123
    This file has been hand-tuned. It is no longer an MLE output.
```

The hash doesn't prevent editing. It makes edits VISIBLE. The distinction
between "inference-derived" and "hand-tuned" is auditable.

### 5.3 `fit_record.json`

Self-contained provenance record. Everything needed to understand and reproduce
the fit:

```json
{
  "model": {
    "path": "models/he2010_london.camdl",
    "hash": "3a7f2c1d"
  },
  "data": {
    "weekly_cases": {
      "path": "data/london_cases.tsv",
      "hash": "f9e2b047",
      "n_observations": 780,
      "time_range": [7.0, 5467.0]
    }
  },
  "fit_config": {
    "path": "fit.toml",
    "hash": "b2c4d6e8",
    "estimated": ["R0", "sigma", "gamma", "rho", "amplitude", "S0", "E0", "I0"],
    "fixed": ["N0", "mu", "k", "psi", "cohort"]
  },
  "method": {
    "algorithm": "IF2",
    "backend": "chain_binomial",
    "dt": 1.0,
    "seed": 1,
    "stages": {
      "scout": { "chains": 8, "particles": 200, "iterations": 20 },
      "refine": { "chains": 4, "particles": 1000, "iterations": 50 },
      "validate": { "chains": 4, "particles": 5000, "iterations": 100 }
    }
  },
  "results": {
    "mle": {
      "R0": 56.82,
      "sigma": 0.0791,
      "gamma": 0.0832,
      "rho": 0.488,
      "amplitude": 0.554,
      "S0": 73151,
      "E0": 127,
      "I0": 127
    },
    "loglik": -3804.9,
    "loglik_sd": 5.2,
    "initial_loglik": -4523.7,
    "ess_at_mle": { "mean": 3842, "min": 1205 },
    "convergence": {
      "chain_agreement_max": 1.03,
      "all_converged": true
    },
    "identifiability": {
      "R0": { "ci_95": [52.1, 62.3], "curvature": 0.23 },
      "sigma": { "ci_95": [0.073, 0.086], "curvature": 1.84 }
    }
  },
  "provenance": {
    "input_hash": "7f2c1d3a",
    "content_hash": "a3c1e890",
    "timestamp": "2026-03-30T14:30:00Z",
    "camdl_version": "0.3.0",
    "wall_time_seconds": 1847
  }
}
```

---

## 6. Stage Details

### 6.1 Scout

```bash
camdl fit scout fit.toml [--seed 1]
```

**Defaults:** 8 chains, 500 particles, 30 iterations, `cooling_fraction = 0.5`
(mild contraction — find basins, don't converge), rw_sd from fit.toml or auto
from bounds (`/20` for log, `/6` for logit).

**Seeded chains:** When `[estimate]` parameters have `start` values, chain 1
starts near those values (jittered by rw_sd). Remaining chains start from random
positions within bounds. Controlled by `start_chains` in `[scout]` (default 1
when any start exists).

**Writes:**

```
<fit_dir>/real/fit_<seed>/scout/
  chain_{1..8}/parameter_traces.tsv
  chain_{1..8}/final_params.toml    ← per-chain winner + [provenance]
  fit_state.toml                    ← inter-stage handoff
  mle_params.toml                   ← winner θ̂ + [provenance]
  final_params.toml                 ← winner θ̂ + [provenance] (run-root)
  chain_evaluations.tsv             ← full loglik-eval score table
  diagnostics.json                  ← structured warnings
  run.json                          ← stage metadata (post-Phase-3)
```

> **v2 layout note.** The `real/fit_<seed>/` wrapper was added in commit
> `5f1e704` (2026-04-18). Pre-v2 paths (`fit/{name}/scout/...`) are stale; no
> fit produced by `cmd_fit_run_v2` writes that shape.

For an interpretation surface — gate verdict, parameter table, filter health —
read `camdl fit summary --format json` (§7.2.2). There is no separate
`<stage>_summary.json` file written; that was a v1 artefact superseded by the
summary command.

#### 6.1.1 Clean-evaluation re-scoring and the compound gate

Picking the winning chain by argmax over IF2's in-run 500-particle PF
log-likelihood is biased by ~tens of nats (Monte Carlo noise gets selected on,
not just signal), and "all Â small" alone fails to catch chains that agree
per-parameter while sitting in different likelihood basins. After IF2 finishes,
scout runs a loglik-evaluation pass:

1. **Candidate.** Each chain contributes one candidate θ̂: the IF2
   final-iteration parameter mean (`iterations.last().param_means`). This is the
   pomp-canonical estimator for IF2 (King, Ionides, Bretó 2016 JSS; Ionides et
   al. 2015 PNAS supplementary materials). Within-chain candidate selection
   (e.g. argmax over a finite candidate pool) was stripped in commit `20d48fe`
   because it introduced selection-bias-on-top-of-bias-fix; see
   `docs/dev/notes/2026-04-27-clean-eval-strip.md` for the audit trail.
2. **Independent re-scoring.** Each candidate is scored with M independent
   high-particle PF replicates. Per-replicate logliks are combined via
   `logmeanexp` (unbiased on the likelihood scale) to a single combined score;
   the SE is the delta-method approximation on the log scale via
   `evidence::logmeanexp_with_se`. Defaults: `n_particles = 4000`,
   `n_replicates = 8`, `combine = LogMeanExp`. Override per-stage with
   `--loglik-eval-particles N` / `--loglik-eval-reps M`.
3. **Cross-chain selection.** The maximum across per-chain clean logliks names
   the overall winner (the chain whose θ̂ is reported as the MLE). NaN chains are
   explicitly skipped; an all-NaN cohort errors rather than silently picking
   chain 0.

The compound scout-convergence gate (read by `camdl fit refine`) passes iff:

- `max(Â) < a_thresh` over per-parameter chain-agreement, **and**
- `Δ_dB < threshold_dB` over the per-chain loglik-eval logliks,

where `Δ_dB = (max − min) · NATS_TO_DB` is the decibans spread, and
`threshold_dB = max(decibans_thresh, 8 · σ_max · NATS_TO_DB)` with
`σ_max = max(chain_eval_ses)`. The SE-aware floor prevents the gate from firing
on noisy chains whose spread is statistically indistinguishable from zero.
Defaults: `a_thresh = 1.01`, `decibans_thresh = 30.0`. Override per-stage with
`--decibans-thresh X`.

Status surfaces the verdict as
`loglik-eval Δ = X dB / threshold Y dB (σ_max=Z) ✓/✗` under scout and refine.
The full per-(chain × candidate) score table is written to
`<stage>/chain_evaluations.tsv` (header:
`chain  candidate  loglik
se  <param₁>  …`) and the run-root winner to
`<stage>/final_params.toml`.

### 6.2 Refine

```bash
# In fit.toml: [stages.refine] init = "from_mle", init_mle = "scout"
camdl fit run fit.toml --stage refine [--seed 1]
```

**Defaults:** 4 chains, 1000 particles, 50 iterations, cooling_fraction=0.95
over 50 iterations.

**Reads:** `scout/fit_state.toml` for start values and rw_sd (resolved from the
`init_mle` toml key).

**Writes:** same shape as scout (above). Use `camdl fit summary --format json`
for the interpretation surface (§7.2.2).

### 6.3 Validate

```bash
# In fit.toml: [stages.validate] init = "from_mle", init_mle = "refine"
camdl fit run fit.toml --stage validate [--seed 1]
```

**Defaults:** 4 chains, 5000 particles, 100 iterations, cooling_fraction=0.95.
Profiles for ALL estimated parameters (embarrassingly parallel). Final pfilter
at MLE with N=10000 for precise loglik and ESS measurement.

**ESS at MLE:** The most informative single diagnostic. Run a clean pfilter at
the MLE and report mean and min ESS across observation times. High ESS (>N/2)
means the model generates trajectories that track the data. Low ESS (<N/4) means
the model can't produce trajectories consistent with the data even at its best
parameters:

```
ESS at MLE: mean=3842, min=1205  ✓ filter is healthy
```

or:

```
ESS at MLE: mean=312, min=8  ✗ filter is degenerate
  Possible causes:
    - Observation model too tight (estimate psi, or increase it)
    - Process noise too low (estimate sigma_se, or increase it)
    - Model structure cannot reproduce observed dynamics
```

**Reads:** `refine/fit_state.toml`.

**Writes:**

```
<fit_dir>/real/fit_<seed>/validate/
  chain_{1..4}/
    parameter_traces.tsv
    final_params.toml
  profiles/
    {param}_profile.tsv       ← one per estimated parameter (when validate
                                 runs profiles; not yet wired in v2)
  fit_state.toml
  mle_params.toml             ← final MLE (provenance-hashed)
  pfilter_loglik.txt          ← precise loglik at MLE
  ess_at_mle.tsv              ← per-observation ESS trace at MLE
  run.json                    ← stage metadata
```

> **v2 layout note.** Same `real/fit_<seed>/` wrapper convention as
> scout/refine; introduced in commit `5f1e704` (2026-04-18).

For an interpretation surface, `camdl fit summary --format json` (§7.2.2) covers
all stages uniformly.

---

## 7. Status and Summary Commands

camdl exposes three single-fit / multi-fit commands with sharp, non-overlapping
responsibilities:

| command                   | answers                                  | scope                     |
| ------------------------- | ---------------------------------------- | ------------------------- |
| `camdl fit table <dir>`   | "what fits live under this tree?"        | cross-fit aggregator      |
| `camdl fit summary <dir>` | "what does this fit say?"                | single-fit interpretation |
| `camdl compare`           | "which of these models predicts better?" | multi-model comparison    |

### 7.1 Summary

```bash
camdl fit summary results/fits/<dir>/
```

Walks the fit dir, checks which stages have completed, reports convergence and
identifiability:

```
fit/he2010/ — He et al. 2010 London measles

  scout:     ✓ complete (8 chains, best loglik -3891.2, 6/8 good)
  refine:    ✓ complete (4 chains, Â < 1.05, loglik -3804.9)
  validate:  ✓ complete (profiles clean, loglik -3804.9 ± 5.2)

  ESS at MLE: mean=3842, min=1205  ✓ filter is healthy

  Estimated (8 parameters):
    R0        = 56.82   rw_sd=5.0    ✓ identified  CI: [52.1, 62.3]
    sigma     = 0.0791  rw_sd=0.005  ✓ identified  CI: [0.073, 0.086]
    gamma     = 0.0832  rw_sd=0.005  ✓ identified  CI: [0.077, 0.090]
    rho       = 0.488   rw_sd=0.02   ✓ identified  CI: [0.45, 0.53]
    amplitude = 0.554   rw_sd=0.03   ✓ identified  CI: [0.49, 0.61]
    S0        = 73151   rw_sd=5000   ✓ identified  (ivp)
    E0        = 127     rw_sd=100    ✓ identified  (ivp)
    I0        = 127     rw_sd=50     ✓ identified  (ivp)

  Fixed (5 parameters):
    N0     = 2462500
    mu     = 0.0000548
    k      = 50.0
    psi    = 0.116
    cohort = 0.0

  Provenance:
    mle_params.toml: ✓ content hash matches (a3c1e890)
    fit_record.json: ✓ input hash 7f2c1d3a

  Next: camdl batch run experiment.toml \
          --params <fit_dir>/real/fit_<seed>/validate/mle_params.toml
```

A stage that ran IF2 + loglik-eval but failed the compound gate (no `run.json`)
is reported with a one-line pointer:

```
fit/he2010/
    scout       ✗ gate failed — see `camdl fit summary fit/he2010`
```

— rather than the pre-Phase-1 lie "(no completed stages found)".

### 7.2 Summary

```bash
camdl fit summary fit/he2010
```

Reads each MLE stage's `fit_state.toml` + `final_params.toml` +
`mle_params.toml` and renders a per-stage interpretation block:

- compound scout-convergence gate verdict (Â + decibans-spread, rendered against
  the _resolved_ gate config persisted in `fit_state.toml` — i.e. the threshold
  the run was actually judged by, not whatever `fit.toml` says at summary-time)
- parameter table: estimated params with Â glyph, IVP marker
- per-chain loglik-eval table with winner row marked
- provenance cross-check — `final_params.toml ↔ mle_params.toml` and
  `fit_state.toml ↔ final_params.toml`. The `final ↔ mle` cross-check guards
  against the GH #16 silent-wrong-answer class on every read.

#### 7.2.1 Output formats

| `--format`       | use case                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------- |
| `text` (default) | terminal block with ANSI colour. Auto-disables under non-TTY; honours `NO_COLOR`.           |
| `json`           | machine-readable, versioned via `schema.version: 1`. The book pipeline reads this directly. |
| `md`             | GitHub-flavoured Markdown. Embeddable in chapters via Quarto's `run_cli`.                   |
| `latex`          | `\begin{tabular}` blocks per stage. No preamble — embed inside an existing document.        |

#### 7.2.2 JSON schema

Top-level shape:

```json
{
  "schema": {"version": 1, "camdl_version": "0.3.0+abc1234"},
  "fit_dir": "fit/he2010",
  "stages": [
    {
      "name": "scout",
      "n_chains": 8,
      "best_loglik": -6235.1,
      "initial_loglik": -7891.0,
      "camdl_version": "0.3.0+abc1234",
      "gate": {
        "max_a_hat": 1.61,
        "max_a_param": "alpha",
        "a_thresh": 1.01,
        "a_passes": false,
        "delta_db": 205.4,
        "threshold_db": 30.0,
        "sigma_max": 2.2,
        "db_passes": false,
        "overall_pass": false,
        "threshold_source": "resolved",
        "resolved_gate": {"a_thresh": 1.01, "decibans_thresh": 30.0},
        "resolved_loglik_eval": {"n_particles": 4000, "n_replicates": 8, "combine": "log_mean_exp"}
      },
      "stage_progression": null,
      "parameters": [
        {"name": "R0", "estimate": 87.67, "chain_agreement": 1.21, "ivp": false},
        ...
      ],
      "chains": [
        {"chain_id": 1, "clean_loglik": -6760.4, "clean_se": 2.4, "is_winner": false},
        ...
      ],
      "provenance": {
        "final_params_matches_mle_params": true,
        "fit_state_winner_matches_final_params": true,
        "stale_camdl_version": null
      },
      "_heuristic": {
        "overall_status": "fail",
        "interpretation": "chains disagree on basin (Â and decibans-spread both fail)"
      }
    }
  ]
}
```

Schema-stability rules:

- **`schema.version` is the contract.** Adding fields is non-breaking; renames /
  removals / type changes bump the version. Consumers that don't recognise
  newly-added fields must skip them.
- **`_heuristic` is advisory.** Strings inside (`overall_status`,
  `interpretation`) may shift across camdl versions even at stable
  `schema.version`. Hard fields outside `_heuristic` (numeric thresholds,
  pass/fail booleans) obey the schema-version contract.
- **`gate.threshold_source`** is `"resolved"` when the persisted resolved gate
  config was found in `fit_state.toml` (the post- Phase-3 case), and
  `"default_fallback"` for legacy fit_state files written before Phase 3 — in
  which case the rendered thresholds come from `GateConfig::default()` and may
  differ from what the run was actually judged against.

#### 7.2.3 `--params-only`

```bash
camdl pfilter model.camdl --data cases.tsv \
    --params <(camdl fit summary --params-only fit/he2010)
```

Emits only the winner θ̂ as a flat parameter TOML — no headers beyond a single
`# camdl ...` comment, no `[provenance]` block, no metadata. By default picks
the _terminal_ completed stage in pipeline order (validate → refine → scout).
Combine with `--stage <stage>` to pick a different stage's winner.

Use case: re-run the pfilter under a higher particle budget or a different RNG
seed at the reported MLE, without hand-stripping metadata from
`final_params.toml`.

#### 7.2.4 `--strict`

Exits non-zero on any provenance mismatch:

- `final_params.toml` ↔ `mle_params.toml` disagree on the shared parameter
  subset (the GH #16 silent-wrong-answer class)
- `fit_state.toml`'s `start_values` diverge from the winner's θ̂ in
  `final_params.toml`

**Auto-enabled when `CI=true` or `CI=1` is set in the environment**, matching
cargo / pytest convention. Without strict mode the cross-checks still render `✗`
glyphs; the binary just exits 0 so interactive use isn't disrupted.

---

## 8. Connection to Experiment System

The inference pipeline's output is the experiment system's input. One shared
format: params.toml.

```toml
# experiment.toml
[experiment]
model = "models/he2010_london.camdl"
params = [
  "<fit_dir>/real/fit_<seed>/validate/mle_params.toml",
] # ← from inference

[config]
backend = "chain_binomial"
seeds = { n = 200 }
# ...
```

The experiment system doesn't know or care that the params came from IF2. It's
just a params.toml. But the provenance hash in the comments lets anyone trace it
back to the fit.

### 8.1 The Complete Pipeline

```
model.camdl + data.tsv + fit.toml
    │
    ├── camdl fit scout    → <fit_dir>/real/fit_<seed>/scout/fit_state.toml
    ├── camdl fit refine   → <fit_dir>/real/fit_<seed>/refine/mle_params.toml
    └── camdl fit validate → <fit_dir>/real/fit_<seed>/validate/mle_params.toml
                                      │
                    ┌─────────────────┤
                    ▼                  ▼
            experiment.toml      voi.toml
            (params = [mle])     (experiment = ...)
                    │                  │
                    ▼                  ▼
            camdl simulate       camdl voi run
            batch                    │
                    │                  ▼
                    ▼              evsi.tsv
            sobol_indices.tsv          │
                    │                  ▼
                    └────────┬─────────┘
                             ▼
                         DECISION
```

### 8.2 What the Experiment System Gains

With inference-derived parameters:

- **Sensitivity analysis at the MLE.** Sobol indices tell you which parameter
  uncertainties matter most for the decision — but only if you're centered on
  the right parameter values. Sensitivity analysis at the prior mean is less
  useful than at the MLE.

- **VOI with informative priors.** The IF2 profiles give approximate posterior
  marginals. These feed into the VOI spec's `prior` fields as Beta/LogNormal
  fits to the profile curves. EVSI with data-informed priors is more realistic
  than with vague priors.

- **Predictive simulation.** `camdl simulate` at the MLE produces the "best
  guess" trajectory. The experiment system's seed ensemble around the MLE
  produces prediction intervals.

---

## 9. Provenance Design

### 9.1 Two Hashes

**Input hash:** identifies the computation. Stored in `fit_record.json`.

```
input_hash = sha256(
    model_ir_bytes +
    sorted(data_file_bytes for each stream) +
    fit_toml_bytes +
    seed +
    camdl_version_string
)[:8]
```

Covers all inputs including the RNG seed. Two runs with different seeds have
different input hashes — the computation is fully specified and reproducible.

**Content hash:** detects manual edits. Stored in `mle_params.toml` comment
header.

```
content_hash = sha256(
    sorted(param_name + "=" + format(param_value) for each parameter)
)[:8]
```

Computed purely from the parameter values in the file. If any value is changed,
the content hash no longer matches.

### 9.2 Overwrite and Cache Semantics

When a fit command runs, it computes the input hash and checks whether the
output directory already contains results with a matching input hash:

```
$ camdl fit scout fit.toml --seed 1

  scout already complete for these inputs (input_hash: 7f2c1d3a).
  Use --force to re-run.
```

**Rules:**

- Input hash match → skip (results are deterministic given inputs)
- Input hash mismatch → run (different inputs = different computation)
- `--force` → always run (bypass cache, overwrite existing results)
- No `--seed` given → generate a seed, record it, include in hash

This ensures the cache is reliable: same inputs always produce the same hash,
and different inputs always produce different hashes. The seed in the hash is
what makes this work — without it, two runs with different seeds would look
identical to the cache.

### 9.3 Content Verification

`camdl fit status` checks whether `mle_params.toml` has been modified since the
fit produced it:

```rust
fn verify_mle_params(path: &str) -> ProvenanceStatus {
    let content = read_file(path);
    let declared_hash = extract_content_hash(&content);
    let current_hash = compute_content_hash(&parse_toml_values(&content));
    
    if declared_hash == current_hash {
        ProvenanceStatus::Valid
    } else {
        ProvenanceStatus::Modified {
            changes: diff_params(declared_hash, current_hash)
        }
    }
}
```

### 9.4 When Provenance Breaks

If the user edits `mle_params.toml`, the content hash is broken. This is not an
error — it's information:

```
mle_params.toml: ⚠ MODIFIED (content hash mismatch)
  R0: 56.82 → 60.0 (manual edit)
  All other parameters: unchanged
  
  This file has been hand-tuned. Inference provenance no longer applies.
  To restore: camdl fit run fit.toml --stage validate
              ([stages.validate] init = "from_mle", init_mle = "refine")
```

The modified file still works as a params.toml everywhere. The provenance status
is advisory, not blocking.

---

## 10. Data File Format

### 10.1 Single-Stream

```tsv
time	cases
7	23
14	67
21	134
28	201
```

`time` in model time units (days for He et al.). One value column named to match
the observation projection output.

**Observation times must be exact multiples of dt.** A time that does not fall
on a step boundary is a hard error:

```
error: observation at t=7.5 is not a multiple of dt=1.0.
  The chain-binomial state only exists at step boundaries.
  Adjust observation times or dt to align.
```

Silent snapping would change which flows accumulate into the projected quantity,
altering the likelihood. This is a user error that must be caught, not papered
over.

### 10.2 Multi-Stream

Each observation stream has its own file. The `[data]` section in fit.toml maps
stream names to files:

```toml
[data]
afp_cases = "data/afp_by_district.tsv"
es_positive = "data/es_results.tsv"
```

### 10.2.1 Incidence vs. Prevalence

Incidence data (event counts accumulated over the interval since the last
observation) and prevalence data (point-in-time compartment counts) project the
simulator state differently. Both are first-class: the observation block's
projection (`incidence(X)`, `prevalence(X)`, or a derived expression like
`B1 + B2`) selects the mode. See `camdl-run-spec.md` §14 for the snapshot-timing
and intervention-ordering rules.

Prevalence and incidence carry different Fisher information about the parameters
— prevalence is more informative about the recovery rate γ (decay shape),
incidence about the transmission rate β (direct flow into I). A fit against
prevalence-only data may have a substantially wider posterior for β than the
same model fit against incidence data on the same trajectory; this is a feature
of the data, not a bug in the fit.

Pair likelihoods with projection types deliberately: NegBinomial or
Poisson-with-reporting for incidence, Binomial or Poisson for prevalence counts,
Beta or Binomial for prevalence fractions. The `fit run` and `pfilter` startup
diagnostic lists the `(projection, likelihood)` pairing for every stream so that
a mismatch is visible before the filter runs.

### 10.3 Missing Data

Rows with `NA` or blank values are skipped (no contribution to log-likelihood at
that time). Missing entire time points: omit the row.

### 10.4 Spatially Indexed Data

For spatial models with per-patch observations:

```tsv
time	patch	cases
30	kano	3
30	lagos	0
30	sokoto	1
60	kano	2
60	lagos	1
```

The `patch` column matches the model's spatial dimension. The particle filter
sums log-likelihoods across patches at each time.

---

## 11. Relationship to Other Specs

```
┌──────────────────────────┐
│  .camdl model            │
│  observations {} block   │ structure, obs model
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  fit.toml                │ what to estimate, what data
│  Inference Workflow Spec  │
└────────────┬─────────────┘
             │ camdl fit scout → refine → validate
             ▼
┌──────────────────────────┐
│  mle_params.toml         │ IS a params.toml
│  fit_record.json         │ provenance
└────────────┬─────────────┘
             │ referenced by
             ▼
┌──────────────────────────┐
│  experiment.toml         │ designs, sweeps, comparisons
│  Experiment Spec v0.5    │ → outputs.tsv
└────────────┬─────────────┘
             │ consumed by
             ▼
┌──────────────────────────┐
│  voi.toml                │ decisions, studies, EVSI
│  VOI Spec v0.1           │ → evsi.tsv
└──────────────────────────┘
```

The dependency chain is one-way. Each spec consumes the outputs of the one above
it. No circular dependencies. The observation model lives in the .camdl file and
is shared by both the fit pipeline (for dmeasure) and the experiment pipeline
(for synthetic observations).
