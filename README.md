# camdl

[![CI](https://github.com/vsbuffalo/camdl/actions/workflows/ci.yml/badge.svg)](https://github.com/vsbuffalo/camdl/actions/workflows/ci.yml)

**Compartmental Model Description Language** — a DSL and toolchain for
stochastic compartmental epidemic models. Write the math, not the code.

An OCaml compiler expands `.camdl` model specifications into a flat JSON
intermediate representation. A Rust backend simulates, fits, and analyzes them.

```
model.camdl ──→ camdlc ──→ model.ir.json
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              camdl simulate  camdl fit  camdl experiment
                    │           │           │
                    ▼           ▼           ▼
              trajectory    mle_params   sobol_indices
                            profiles     evsi
```

## Capabilities

| Domain                | What camdl does                                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Modelling**         | Compartments, stratification (age, space, risk), contact matrices, Erlang staging, forcing functions, interventions, events, balance constraints, scenarios                |
| **Simulation**        | Gillespie SSA, tau-leap, chain-binomial (Euler-multinomial), ODE (RK4). Extra-demographic stochasticity via `overdispersed()`. Deterministic flows via `deterministic()`. |
| **Inference**         | IF2 (MLE), PGAS with NUTS (Bayesian posterior), bootstrap particle filter, 1D/2D profiles. Source-to-source autodiff: compiler emits gradient expressions, enabling HMC. |
| **Fitting workflow**  | `camdl fit scout → refine → validate → pgas` pipeline driven by `fit.toml`. IF2 finds the MLE, PGAS characterizes the posterior with exact complete-data likelihood + NUTS gradient proposals. Parallel chains, random starts, R-hat/ESS diagnostics, posterior trajectory output. |
| **Experiments**       | Multi-scenario seed ensembles, Sobol sensitivity analysis, parameter sweeps. Content-addressable output with caching.                                                     |
| **Decision analysis** | Value of Information (EVSI) via `camdl voi run`.                                                                                                                          |

---

## Build

```bash
make build       # build both OCaml and Rust
make install     # copy binaries to ~/.local/bin
make test        # full test suite (28 tests)
```

## Quick start

```bash
# Simulate an SIR model
camdl simulate ocaml/golden/sir_basic.camdl \
  --param beta=0.3 --param gamma=0.1 --param N0=1000 --param I0=10 \
  --output traj.tsv

# Fit a model to data
camdl fit scout fit.toml
camdl fit refine fit.toml --starts-from fit/output/scout/
camdl fit validate fit.toml --starts-from fit/output/refine/
camdl fit status fit.toml

# Run an experiment
camdl experiment run experiment.toml
camdl experiment analyze experiment.toml
```

---

## The camdl language

A `.camdl` file defines model structure. Parameter values are supplied at
runtime via `--param` flags or `--params file.toml`.

### SIR

```
time_unit = 'days

compartments { S, I, R }
let N = S + I + R

parameters {
  beta  : rate     in [0.001, 2.0]
  gamma : rate     in [0.001, 1.0]
  N0    : count    in [100, 100000]
  I0    : count    in [1, 1000]
}

transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

simulate {
  from = 0 'days
  to   = 120 'days
}
```

### Age-structured SEIR with contact matrix

Stratification expands compartments and transitions at compile time. The IR
contains only flat compartments and fully-resolved rates.

```
compartments { S, E, I, R }
stratify(by = age, values = [child, adult])

let N_local[a in age] = S[a] + E[a] + I[a] + R[a]

tables {
  C_age : age × age = [[12.0, 4.0], [4.0, 8.0]]
}

transitions {
  infection[a in age] : S[a] --> E[a]
    @ beta * S[a] * sum(b in age, C_age[a, b] * I[b] / N_local[b])

  progression[a in age] : E[a] --> I[a]  @ sigma * E[a]
  recovery[a in age]    : I[a] --> R[a]  @ gamma * I[a]
}
```

### Seasonal forcing and stochastic rates

```
forcing {
  seasonal : sinusoidal {
    amplitude = amplitude
    period    = 365.25
    phase     = 0.0
    baseline  = 1.0
  }
}

let beta = R0 * gamma * seasonal(t)

transitions {
  infection : S --> E  @ overdispersed(beta * S * I / N, sigma_se)
  recovery  : I --> R  @ gamma * I
  birth     : --> S    @ deterministic(mu * N)
}

observations {
  cases : {
    projected  = incidence(recovery)
    every      = 7 'days
    likelihood = neg_binomial(mean = rho * projected, r = k)
  }
}
```

`overdispersed(rate, sigma)` adds extra-demographic stochasticity
(Gamma-Poisson, He et al. 2010). `deterministic(rate)` uses rounding instead of
Poisson draws. `t` is the current simulation time.

Forcing function types: `sinusoidal`, `periodic` (with range-based schedules
`on = [7:100, 115:199]`), `piecewise`, `interpolated` (cubic spline or linear).

### More language features

- **Stratification**: `stratify(by = dim, values = [...])` with partial
  `only = [COMP]`
- **Indexed transitions**: `[a in age]` binds index variables
- **Erlang staging**: `consecutive(dim)` for linear chain trick
- **Guards**: `where src != dst` for compile-time filtering
- **Coupling**: `coupling(age) = C_age` for contact-matrix mixing
- **Let bindings**: `let name[i] = expr` (inlined at use sites)
- **Math functions**: `exp`, `log`, `sqrt`, `abs`, `floor`, `ceil`, `mod`
- **Interventions**: `transfer(fraction=..., from=..., to=...)` with `at [...]`
  or `every E from F to T` schedules
- **Scenarios**: named parameter/intervention presets for comparison

Full reference: [`docs/camdl-language-spec.md`](docs/camdl-language-spec.md)

---

## Simulation

```bash
camdl simulate MODEL [--backend gillespie|tau_leap|chain_binomial|ode]
    [--dt DT] [--seed N] [--params P.toml] [--param K=V] [--output FILE]
```

| Backend        | Flag                                | Notes                                           |
| -------------- | ----------------------------------- | ----------------------------------------------- |
| Chain-binomial | `--backend chain_binomial --dt 1.0` | Euler-multinomial (multinomial competing risks); default — matches `camdl fit` |
| Gillespie SSA  | `--backend gillespie`               | Exact stochastic                                |
| Tau-leap       | `--backend tau_leap --dt 0.5`       | Poisson approximation                           |
| ODE (RK4)      | `--backend ode --dt 0.1`            | Deterministic                                   |

Same seed + same backend = identical trajectory (Common Random Numbers).
`overdispersed()` requires tau-leap or chain-binomial. The backend capability
system enforces this at dispatch time.

---

## Inference

### Particle filter

```bash
camdl pfilter MODEL --params P.toml --data cases.tsv \
    --particles 5000 --dt 1 --seed 1 \
    --flow recovery --trace diag.tsv --output ll.txt
```

Bootstrap particle filter with systematic resampling. Reports log-likelihood,
per-observation ESS, and one-step-ahead prediction quantiles.
`--save-final-state particles.tsv` exports the particle ensemble for prediction
workflows.

### IF2 (iterated filtering)

```bash
# Explicit rw_sd — the list IS the partition
camdl if2 MODEL --params P.toml --data cases.tsv \
    --rw-sd "R0=5,sigma=0.01" --regime refine --flow recovery

# Auto rw_sd from parameter bounds
camdl if2 MODEL --params P.toml --data cases.tsv \
    --rw-sd auto --fixed "N0,mu,k" --regime scout
```

Multi-chain IF2 with per-chain indicatif progress bars, Rhat convergence
diagnostics, and regime presets (scout/refine/validate). Auto rw_sd computes
`(hi-lo)/6` on the transformed scale from parameter bounds.

### Profile likelihood

```bash
camdl profile MODEL --params P.toml --data cases.tsv \
    --focal R0 --grid "20,30,40,50,60,70,80" \
    --rw-sd "sigma=0.01,gamma=0.01" \
    --output profile_R0.tsv --parallel 8
```

Parallel 1D and 2D profile likelihoods with indicatif progress bars.

### Fit workflow

The structured fitting pipeline, driven by `fit.toml`:

```bash
# MLE via IF2
camdl fit scout    fit.toml                          # random starts, landscape discovery
camdl fit refine   fit.toml --starts-from scout/     # convergent IF2 from best scouts
camdl fit validate fit.toml --starts-from refine/    # profiles + precise pfilter at MLE

# Bayesian posterior via PGAS
camdl fit pgas     fit.toml --starts-from validate/  # Gibbs sampler with NUTS proposals
camdl fit pgas     fit.toml                          # random starts (no prior stage)
camdl fit pgas     fit.toml --no-nuts                # fall back to MH-within-Gibbs

camdl fit status   fit.toml                          # colored summary of all stages
```

**Scout → Refine → Validate** finds the MLE via IF2 (iterated filtering).
**PGAS** (Particle Gibbs with Ancestor Sampling) characterizes the Bayesian
posterior. Each Gibbs sweep alternates exact parameter updates (complete-data
log-likelihood, no PF noise) with CSMC-AS trajectory updates. NUTS
(No-U-Turn Sampler) proposes parameters jointly using analytical gradients
from the compiler's source-to-source autodiff.

PGAS features: parallel chains with random starts, Robbins-Monro adaptive
proposals, diagonal mass matrix adaptation, R-hat/ESS convergence
diagnostics, per-sweep trajectory renewal tracking, posterior trajectory
output (`n_trajectories` in fit.toml).

**Out-of-sample validation:** Add `[holdout]` to fit.toml with holdout data
files. Scout/refine never see holdout data. Validate reports separate
train/holdout logliks. Use `camdl data split cases.tsv --at-time 5474` to
produce train/holdout files.

Specification: [`docs/camdl-inference-spec.md`](docs/camdl-inference-spec.md)

---

## Experiments

```bash
camdl experiment run experiment.toml       # parallel seed ensembles
camdl experiment status experiment.toml    # check progress
camdl experiment summarize experiment.toml # aggregate trajectories
camdl experiment analyze experiment.toml   # Sobol sensitivity indices
```

Multi-scenario parameter sweeps with content-addressable output and caching.
Sobol first-order and total-effect sensitivity indices for any output metric.

Specification: [`docs/camdl-experiment-spec.md`](docs/camdl-experiment-spec.md)

### Value of Information

```bash
camdl voi run voi.toml
```

Expected Value of Sample Information (EVSI) for study design decisions. Combines
experiment outputs with decision problems, prior distributions, and utility
functions.

Specification: [`docs/camdl-voi-spec.md`](docs/camdl-voi-spec.md)

---

## Compiler

### Compile

```bash
camdlc model.camdl > model.ir.json
```

### Check

```bash
$ camdlc check model.camdl

model
  compartments   4 base × 2 age = 8 expanded
  transitions     3 base → 6 expanded
  parameters      3 declared
  ✓ no errors, 0 warnings
```

Errors show source locations:

```
error[E100]: undeclared name 'R'

  ┌─ model.camdl:7:37
7 │   recovery  : I --> R @ gamma * I * R
  │                                    ^
  = hint: check spelling, or add a declaration
```

### Inspect

```bash
camdlc inspect model.camdl --transitions
camdlc inspect model.camdl --transition infection_child
camdlc inspect model.camdl --summary
```

---

## Testing

```bash
make test          # all OCaml + Rust + integration tests (28 tests)
make build         # build both languages
```

CI runs on every push: OCaml compiler tests, Rust unit tests, clippy (warnings =
errors), golden file regeneration + diff check, and the full integration suite.

---

## Documentation

| Document                                                         | Contents                             |
| ---------------------------------------------------------------- | ------------------------------------ |
| [`docs/camdl-language-spec.md`](docs/camdl-language-spec.md)     | Full DSL reference                   |
| [`docs/camdl-data-spec.md`](docs/camdl-data-spec.md)             | IR schema and data model             |
| [`docs/camdl-inference-spec.md`](docs/camdl-inference-spec.md)   | Fitting workflow specification       |
| [`docs/camdl-experiment-spec.md`](docs/camdl-experiment-spec.md) | Experiment system specification      |
| [`docs/camdl-voi-spec.md`](docs/camdl-voi-spec.md)               | Value of Information specification   |
| [`docs/inference.md`](docs/inference.md)                         | Inference guide (PF, IF2, PGAS, NUTS)|
| [`docs/runtimes.md`](docs/runtimes.md)                           | Simulation backend details           |
| [`docs/user-features.md`](docs/user-features.md)                 | Feature catalog with pomp comparison |
| [`docs/intro.md`](docs/intro.md)                                 | DSL tutorial                         |
| [`docs/debugging.md`](docs/debugging.md)                         | Debugging with `camdl eval`          |

---

## Repository layout

```
bin/camdl              Wrapper script (routes to camdlc or camdl-sim)
ocaml/
  lib/compiler/        Lexer, parser, expander (AST → flat IR)
  lib/ir/              OCaml IR types + serialization
  bin/camdlc.ml        Compiler CLI
  golden/              .camdl fixtures + compiled .ir.json
  test/                Compiler unit + golden tests
rust/
  crates/ir/           IR types + serde
  crates/sim/          Simulation backends + propensity evaluator
                       + inference (PF, IF2, PGAS, NUTS, PMMH, obs_loglik)
  crates/observe/      Observation model (likelihood sampling/scoring)
  crates/io/           TSV read/write
  crates/cli/          CLI: simulate, pfilter, if2, profile, fit,
                       experiment, voi, eval, serve
ir/
  schema.json          IR schema (contract between OCaml and Rust)
  golden/              Canonical IR files (Rust test surface)
docs/                  Specifications and guides
examples/              Complete worked examples with params + data
benches/               Criterion benchmarks + performance lab notebook
```

## Architecture

The **IR is fully expanded**: the OCaml compiler performs all stratification,
coupling expansion, and let-binding inlining. The Rust backend sees only flat
compartments, transitions, and expression ASTs.

The **expression language** is pure and first-order:
`Const | Param | Pop | PopSum | Time | BinOp | UnOp | Cond | TimeFunc | TableLookup`.
Evaluated in bounded time at each simulation step.

**Common Random Numbers**: same seed → identical trajectory. Used for
counterfactual scenario comparisons where pre-intervention trajectories are
byte-identical.
