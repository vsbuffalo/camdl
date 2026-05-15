# camdl

[![CI](https://github.com/vsbuffalo/camdl/actions/workflows/ci.yml/badge.svg)](https://github.com/vsbuffalo/camdl/actions/workflows/ci.yml)

**Compartmental Model Description Language** — a DSL and toolchain for
stochastic compartmental epidemic models. Write the math, not the code.

An OCaml compiler expands `.camdl` model specifications into a flat JSON
intermediate representation. A Rust backend simulates, fits, and analyzes them.

```
model.camdl ──→ camdlc ──→ model.ir.json
                                │
          ┌──────────┬──────────┼──────────┐
          ▼          ▼          ▼          ▼
    camdl simulate  camdl fit  camdl batch  camdl compare
          │          │          │           │
          ▼          ▼          ▼           ▼
    trajectory  mle_params   sweep/CAS   elpd table
                profiles     manifest
```

## Capabilities

| Domain                | What camdl does                                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Modelling**         | Compartments, stratification (age, space, risk), contact matrices, Erlang staging, forcing functions, interventions, events, balance constraints, scenarios                |
| **Simulation**        | Gillespie SSA, tau-leap, chain-binomial (Euler-multinomial), ODE (RK4). Extra-demographic stochasticity via `overdispersed()`. Deterministic flows via `deterministic()`. |
| **Inference**         | IF2 (MLE), PGAS with NUTS (Bayesian posterior), bootstrap particle filter, 1D/2D profiles. Source-to-source autodiff: compiler emits gradient expressions, enabling HMC. |
| **Fitting workflow**  | Declarative `fit.toml` (named stages → `camdl fit run`). IF2 finds the MLE; PGAS+NUTS characterises the Bayesian posterior with exact complete-data likelihood + analytical gradients from the compiler. Mandatory convergence gates between stages, Richardson dt-convergence audit after every fit, content-addressable provenance. |
| **Experiments**       | Multi-scenario seed ensembles, Sobol sensitivity analysis, parameter sweeps. Content-addressable output with caching.                                                     |
| **Model comparison**  | Prequential scoring (elpd, CRPS, PIT) + paired Δ table via `camdl compare`. |

---

## Install

### Quick install (Linux / macOS)

For a fresh machine, the `install.sh` script at the repo root installs
both toolchains (OCaml ≥ 5.2 via opam, Rust stable via rustup),
fetches OCaml package dependencies, and runs `make build && make
install`:

```bash
./install.sh
```

It's idempotent — safe to re-run — and uses your system package
manager (apt / dnf / yum / pacman / zypper on Linux, Homebrew on
macOS, installing brew + Xcode CLT if absent). Override the OCaml
switch version with `OCAML_SWITCH_VERSION=5.2.1 ./install.sh`.

If you'd rather wire the toolchain by hand, follow the manual steps
below.

### Prerequisites

camdl has two language runtimes. You need both available before
`make build` will work:

- **OCaml ≥ 5.2 + opam.** macOS: `brew install opam`. Linux: see
  [opam.ocaml.org/doc/Install.html](https://opam.ocaml.org/doc/Install.html).
  Then create a switch:

  ```bash
  opam init -y                    # first-time only
  opam switch create 5.2.0        # match what CI uses
  eval $(opam env)
  ```

- **Rust stable** via rustup ([rustup.rs](https://rustup.rs/)):
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`,
  then `rustup default stable`.

- **Make + git + python3** (only needed for `make update-golden` and the
  integration test driver — usually already installed).

Once-after-cloning, install the OCaml package dependencies declared in
`ocaml/*.opam`:

```bash
cd ocaml
opam install . --deps-only --with-test --yes
cd ..
```

This fetches `dune`, `menhir`, `yojson`, `fmt`, `alcotest`, and the
qcheck stack. Skipping it produces errors like `Library "yojson" not
found` or `Program menhir not found in the tree or in PATH` — those
mean the opam install step hasn't run, not that the build is broken.

### Build

```bash
make build       # builds both OCaml and Rust
make install     # copies camdl + camdlc to ~/.local/bin
make test        # OCaml + Rust + integration (~800 tests in this repo)
```

`make install` is required after every rebuild — `camdl` checks the
on-PATH `camdlc` hash matches its own and refuses to run on a mismatch.
The post-install message warns if another `camdl` (e.g. a leftover
`cargo install`) is shadowing on PATH.

Make sure `~/.local/bin` is on your PATH; on most shells that means
adding `export PATH="$HOME/.local/bin:$PATH"` to `~/.zshrc` or
`~/.bashrc`.

## Quick start

```bash
# Survey the likelihood surface before fitting
camdl survey he2010_london.camdl --n-points 500 --render

# Simulate an SIR model
camdl simulate ocaml/golden/sir_basic.camdl \
  --param beta=0.3 --param gamma=0.1 --param N0=1000 --param I0=10 \
  --output traj.tsv

# Fit a model to data — single declarative pipeline
camdl fit run fit.toml                    # all stages in order
camdl fit run fit.toml --stage scout      # one stage
camdl fit status fit.toml                 # colored summary

# Run a parameter sweep / scenario batch
camdl batch run sweep.toml
camdl batch status sweep.toml
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
dimensions { age = [child, adult] }
stratify(by = age)

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

Forcing function types: `sinusoidal`, `periodic` (range-based schedules
`on = [7:100, 115:199]`), `piecewise`, `interpolated` (linear or cubic
spline), `fourier` (truncated Fourier series), `periodic_spline`
(periodic B-spline via de Boor — cross-validated bit-identical against
pomp 6.4 and `scipy.interpolate.BSpline`).

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
| Gillespie SSA  | `--backend gillespie`               | Exact stochastic; default                       |
| Tau-leap       | `--backend tau_leap --dt 0.5`       | Poisson approximation                           |
| Chain-binomial | `--backend chain_binomial --dt 1.0` | Euler-multinomial (multinomial competing risks) |
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

The fitting pipeline is one declarative file (`fit.toml`) driven by
`camdl fit run`. Stages are named and chained inside the TOML; the CLI
runs them all by default, or one at a time via `--stage`:

```bash
camdl fit run fit.toml                    # all stages, in order
camdl fit run fit.toml --stage scout      # one stage only
camdl fit run fit.toml --resume --stage pgas  # extend a completed PGAS run
camdl fit status fit.toml                 # colored summary of all stages
camdl fit summary <fit-dir>               # Â / gate verdict / MLE table
```

Stage entries declare `algorithm` + `backend` explicitly:

```toml
[stages.scout]
algorithm   = "if2"            # iterated filtering MLE
backend     = "chain_binomial"
init_method = "lhs"            # Latin-hypercube starts (scale-aware)

[stages.refine]
algorithm   = "pgas"           # Bayesian posterior via PGAS+NUTS
backend     = "chain_binomial"
starts_from = "scout"

[stages.validate]
algorithm  = "pfilter"         # held-out predictive log-likelihood
particles  = 4000
replicates = 8
```

`camdl fit methods` lists the supported `(algorithm, backend)` pairs.

**IF2** ([Ionides et al. 2015](https://doi.org/10.1073/pnas.1410597112))
finds the MLE via iterated filtering with cooled perturbations.

**PGAS+NUTS** ([Lindsten et al. 2014](https://jmlr.org/papers/v15/lindsten14a.html);
[Hoffman & Gelman 2014](https://jmlr.org/papers/v15/hoffman14a.html))
characterises the Bayesian posterior. Each Gibbs sweep alternates exact
parameter updates (complete-data log-likelihood — no PF noise) with
CSMC-AS trajectory updates. NUTS proposes parameters jointly using
**analytical gradients** the OCaml compiler emits source-to-source as
`rate_grad` IR fields. No autodiff tape, no JAX dependency.

PGAS features: parallel chains, Robbins-Monro adaptive MH, diagonal
mass-matrix adaptation, R-hat/ESS convergence diagnostics, per-sweep
trajectory renewal tracking, posterior trajectory output
(`n_trajectories` in fit.toml).

**Convergence gates** sit between stages and fail the pipeline rather
than passing a bad fit through. Scout's gate is two-legged: per-parameter
chain agreement (Gelman–Rubin-style on IF2 per-iteration parameter means)
plus a decibans-spread check on chain-level log-likelihoods at high
particle count. Both legs must pass.

**Richardson dt-convergence check** runs after every fit: re-evaluate the
loglik on a halving ladder of integrator steps (`dt`, `dt/2`, `dt/4`)
and refuse to bless a fit whose likelihood is still drifting. Catches
integration-step pathology that synthetic-recovery tests miss (the same
`dt` on both sides cancels the bias).

**Out-of-sample validation:** add `[holdout]` to fit.toml with holdout
data files. Scout / refine never see holdout data; validate reports
separate train / holdout logliks. Use `camdl data split cases.tsv
--at-time 5474` to produce train/holdout files.

Specification: [`docs/camdl-inference-spec.md`](docs/camdl-inference-spec.md)

---

## Batch sweeps

```bash
camdl batch run sweep.toml        # multi-scenario × multi-seed sweep
camdl batch run sweep.toml --dry-run   # preview resolved sweep grid
camdl batch status sweep.toml     # completion / live file count
```

Scenario × sweep-point × seed cartesian products with content-addressable
output in `./results/sims/`. Re-runs skip cached results automatically;
`--force` re-runs anyway. `[design]` blocks in `sweep.toml` generate
Sobol, LHS, or random samples for sensitivity analysis — downstream
analysis (Sobol indices, etc.) reads the resulting `outputs.tsv` from
any tool you prefer.

Specification: [`docs/camdl-run-spec.md`](docs/camdl-run-spec.md)

## Model comparison

```bash
camdl compare fits/a/pfilter fits/b/pfilter --baseline a
```

Prequential out-of-sample scoring (elpd, CRPS, PIT coverage, paired
Δelpd + se, E_T = exp(Δelpd) Bayes factor). Consumes `prequential.json`
written by `fit run` pfilter stages or `camdl pfilter --save-prequential`.

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
make test          # OCaml + Rust + integration (~800 tests in this repo)
make build         # build both languages
```

CI runs on every push: OCaml compiler tests, Rust unit tests, clippy
(warnings = errors), golden file regeneration + diff check, and the
full integration suite.

**External oracles run as CI gates.** Where camdl overlaps with reference
implementations, regression matters: the periodic-B-spline forcing
agrees with both pomp 6.4 (R) and `scipy.interpolate.BSpline` (Python)
within 10⁻¹² across a 200-point grid; the He et al. (2010) London
measles MLE recovers pomp's published values within particle-filter
Monte Carlo error; the bare SIR final-size hits the
Kermack–McKendrick closed form. CI blocks merges if camdl drifts.

---

## Documentation

| Document                                                         | Contents                             |
| ---------------------------------------------------------------- | ------------------------------------ |
| [`docs/camdl-language-spec.md`](docs/camdl-language-spec.md)     | Full DSL reference                   |
| [`docs/camdl-data-spec.md`](docs/camdl-data-spec.md)             | IR schema and data model             |
| [`docs/camdl-inference-spec.md`](docs/camdl-inference-spec.md)   | Fitting workflow specification       |
| [`docs/camdl-run-spec.md`](docs/camdl-run-spec.md)               | Run-system / batch / CAS specification |
| [`docs/inference.md`](docs/inference.md)                         | Inference guide (PF, IF2, PGAS, NUTS)|
| [`docs/runtimes.md`](docs/runtimes.md)                           | Simulation backend details           |
| [`docs/user-features.md`](docs/user-features.md)                 | Feature catalog with pomp comparison |
| [`docs/intro.md`](docs/intro.md)                                 | DSL tutorial                         |
| [`docs/debugging.md`](docs/debugging.md)                         | Debugging with `camdl eval`          |
| [`AGENTS.md`](AGENTS.md)                                         | Briefing for AI coding agents working with camdl |

**Using camdl from a downstream project with an AI coding agent?** Start
with [`AGENTS.md`](AGENTS.md). It covers the canonical workflow, error /
diagnostic interpretation, when to stop and ask the human, and a
shallow-clone recipe for pinning the docs locally so the agent can read the
language spec offline at version-matched cost (~5 MB):

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/vsbuffalo/camdl .camdl-source
cd .camdl-source && git sparse-checkout set docs ocaml/golden && cd ..
```

Add `.camdl-source/` to your project's `.gitignore`; sync with
`git -C .camdl-source pull`.

---

## Repository layout

```
ocaml/
  lib/compiler/        Lexer, parser, expander (AST → flat IR)
  lib/ir/              OCaml IR types + serialization (incl.
                       source-to-source autodiff for rate gradients)
  bin/camdlc.ml        Compiler CLI
  golden/              .camdl fixtures + compiled .ir.json
  test/                Compiler unit + golden tests
rust/
  crates/ir/           IR types + serde + envelope (version handshake)
  crates/sim/          Simulation backends (Gillespie, tau-leap,
                       chain-binomial, ODE) + propensity evaluator
                       + inference (PF, IF2, PGAS+NUTS, PMMH,
                       obs_loglik, prequential)
  crates/io/           TSV read/write
  crates/cli/          camdl: simulate, batch, fit, pfilter, if2,
                       profile, survey, eval, data, list, show, cat,
                       compare, label, compile, check, inspect
ir/
  VERSION              Canonical IR schema version (read at compile time
                       by Rust via include_str!; emitted by OCaml in
                       every IR envelope)
  schema.json          IR schema documentation
  golden/              Canonical IR files (Rust test surface)
docs/                  Specifications and guides
examples/              Complete worked examples with params + data
benches/               Criterion benchmarks + performance lab notebook
```

## Architecture

The **IR is fully expanded**: the OCaml compiler performs all
stratification, coupling expansion, and let-binding inlining. The Rust
backend sees only flat compartments, transitions, and expression ASTs.

The **expression language** is pure, total, and first-order:
`Const | Param | Pop | PopSum | Time | Dt | BinOp | UnOp | Cond | TimeFunc | TableLookup | Projected`.
Evaluated in bounded time at each simulation step. Same property that
makes dim-checking tractable also makes source-to-source autodiff a
~30-line OCaml pattern match — exact ∇ log ℒ at one expression-tree
walk per step, no autodiff tape.

The **IR contract** is enforced via a version envelope
(`{ ir_version, validated_by, model }`) — Rust's deserializer rejects
mismatched schemas at the boundary, so OCaml/Rust drift fails CI rather
than producing wrong-but-parseable simulations.

**Common Random Numbers**: same seed → identical trajectory. Used for
counterfactual scenario comparisons where pre-intervention trajectories
are byte-identical.

**Strict-mode runtime.** Rate-evaluation degeneracies (division by zero,
NaN/Inf from `Pow`, sqrt of negative, binomial overshoot) produce typed
errors by default — `SimError::NumericalCollapse` and `NegativeCount`.
Inference layers catch per-particle-recoverable errors and convert to
−Inf log-likelihood for the offending particle (resampling kills it,
the chain continues). Forward simulation halts. Pass
`--allow-degenerate-rates` to restore the legacy silent-zero on rare
legitimate cases.
