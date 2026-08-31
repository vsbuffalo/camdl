# camdl

[![CI](https://github.com/vsbuffalo/camdl/actions/workflows/ci.yml/badge.svg)](https://github.com/vsbuffalo/camdl/actions/workflows/ci.yml)
[![Compiler](https://github.com/vsbuffalo/camdl/actions/workflows/compiler.yml/badge.svg)](https://github.com/vsbuffalo/camdl/actions/workflows/compiler.yml)
[![Inference](https://github.com/vsbuffalo/camdl/actions/workflows/inference.yml/badge.svg)](https://github.com/vsbuffalo/camdl/actions/workflows/inference.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](VERSIONING.md)


camdl is a small domain-specific language (DSL), simulation runtime, and
statistical inference stack for compartmental infectious disease models. The
goal of camdl is to make writing compartmental models easy --- the mechanics of
a model are expressed in the camdl DSL just as you'd write the math on a
whiteboard. Inspired by Stan, camdl supports multiple runtime backends (ODE,
chain-binomial, Gillespie) and inference methods (iterated filtering, Particle
Marginal Metropolis--Hastings, Particle Gibbs with Ancestral Sampling).

Each model is written in terms of compartments, transitions, rates,
observations, and fitting configuration. Each model is then run through a
compiler that runs extensive checks and validations, catching modeling mistakes
like unit and dimensional mismatches before the model is even run. The camdl
compiler also optimizes the model using tricks like loop-invariant code motion
and binding caches so that it will run faster during simulations and inference.
All of this happens under the hood, automatically --- the researcher does not
need to run these extra steps themselves. As scientific computational workflows
are increasingly done by coding agents, camdl is explicitly forward-looking:
the compiler helps ensure the model agents write is sound, and camdl has
doc-tested documentation integrated into command-line tooling for maximal
discoverability of never-stale info for agents fitting models with camdl.

In a well-posed infectious disease model, the observation process is part of
the model itself --- in camdl, the link between input data and the model's
latent process is linked declaratively in an `observations { }` block, which
supports multiple observation streams (e.g. time-series age-stratified
incidence, *and* environmental surveillance). The mapping between input data
and the observation likelihood is declarative --- camdl automatically takes
care of everything else.

A central design goal of camdl is to make fitting multiple model variants and
model comparisons easy and reproducible. Every camdl run is fully provenanced
and stored in an *input-addressed storage* system, that ensures reproducibility
and automatically caches to prevent accidental overwrites of expensive runs.

Developed at the [Institute for Disease Modeling](https://www.idmod.org/) (IDM),
Gates Foundation.

## A Simple SIR camdl Model

Here's a simple SIR model in camdl:

```camdl
time_unit = 'days

compartments { S, I, R }
let N = S + I + R

parameters {
  #' transmission rate (contacts × per-contact transmission probability)
  #' @symbol β
  beta  : rate  in [0.001, 2.0]
  #' recovery rate — the mean infectious period is 1/γ
  #' @symbol γ
  gamma : rate  in [0.001, 1.0]
}

transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * I
}

init {
  S = 999
  I = 1
}

simulate {
  from = 0 'days
  to = 120 'days
}
```

This is all that is needed to simulate an SIR model with camdl; note that camdl
supports optional [roxygen](https://roxygen2.r-lib.org/)-style comment
documentation (the lines starting with `#'`) embedded in the model, so parameter 
descriptions are automatically stored alongside the model and surface in 
downstream commands like `camdl fit summary` to remind users what's what,
as well as can be used in automatic plotting, etc.

The dimensions in camdl are real types --- `rate` is `time⁻¹`, `S`, `I` are
counts --- which ensures models are valid at their core. These parameter types
also allow camdl's inference stack to automatically transform variables to the
appropriate scale, e.g. probabilities are fit on a logit scale, rates on a
log-scale, etc. After drafting a model like this, users/agents can optionally
type-check it:

```bash
camdl check sir.camdl  # compiles + dimension-checks
```

Had a researcher or coding agent accidentally dropped the `/ N` that makes
transmission per-capita (a classic modelling slip), camdl would reject it at
compile time instead of silently simulating the wrong dynamics:

```text
error[E300]: transition 'infection' rate has wrong dimension
= note: rate = ((beta * S) * I)
```

All of these checks are done automatically though each time a user runs a camdl
command on a `.camdl` file. More often a user would just rely on this automatic
checking happening behind the scenes when they run a command like `camdl
simulate`:

```console
$ camdl simulate sir.camdl --param beta=0.3 --param gamma=0.1
compiling sir.camdl...
 compiled sir.camdl   1.9KB IR in 0.0s (6.1× source)
simulate · chain_binomial: running …
   stored ./results/sims/sir-c6bf78fe/…/seed_1-06cbd6b3
          camdl cat 5ed416f2…
```

Every run is stored under `results/` keyed by its exact inputs (model, config,
parameters, scenario, seed), so an identical re-run is instant and the
`camdl cat <id>` line retrieves the trajectory.

![camdl syntax mirrors the math you'd write on a whiteboard](docs/assets/math-vs-dsl.png)

_The age-stratified force of infection in two notations — standard math (top)
and camdl (bottom). The target-age index `a`, the source-age index `b`, and the
shared `age` dimension are colour-matched across both._

## Fit it to data

Add priors and an observation model, and the same file is ready to fit. The
observation block says how the latent epidemic maps to what you actually
measured — here, weekly case counts that are a reported fraction of incidence,
with negative-binomial noise:

```camdl
parameters {
  beta  : rate         in [0.001, 2.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.001, 1.0]  ~ half_normal(sigma = 0.3)
  rho   : probability  in [0.0, 1.0]    ~ beta(alpha = 2.0, beta = 5.0)   # reporting fraction
  k     : real         in [0.1, 100.0]                                    # NB dispersion
}

observations {
  cases {
    columns   { time : time, cases : count }
    projected = incidence(infection)
    cases     ~ neg_binomial(mean = rho * projected, r = k)
  }
}
```

A `fit.toml` declares the data, what to estimate, and the fitting stages:

```toml
[model]
camdl = "sir.camdl"

[data]
file = "cases.tsv" # a `time` column and a `cases` column

[estimate.beta]
bounds = [0.001, 2.0]
start = 0.3
[estimate.gamma]
bounds = [0.001, 1.0]
start = 0.1

[fixed]
rho = 0.5
k = 10

[stages.fit]
algorithm = "if2" # iterated filtering → MLE
backend = "chain_binomial"
chains = 4
particles = 1000
iterations = 50
cooling = 0.7
```

```bash
camdl fit run fit.toml --seed 1
```

The fit lands in a content-addressed store keyed by its exact inputs (model,
data, config, seed), so re-running an identical fit is instant and every result
is traceable back to what produced it. Browse it with `camdl list --kind fit`,
`camdl fit summary <dir>`, `camdl cat <id>`. Swap the stage to
`algorithm = "pgas"` for a full Bayesian posterior with NUTS, or chain stages
(IF2 to find the mode, PGAS to characterise the posterior around it).

## Why camdl

- **Dimensions are types.** Every parameter carries a unit (`rate`, `count`,
  `probability`, …); every rate expression is dimension-checked at compile time.
  The missing-`/N` bug above is a compile error with a source location, not a
  plausible-looking wrong answer — and for a tool meant to inform public-health
  decisions, that distinction is the point.
- **Three backends, three inference methods — no silent gaps.** Forward:
  Gillespie SSA, chain-binomial (Euler-multinomial), ODE (RK4, or adaptive
  RK4(5)). Inference: iterated filtering (IF2, MLE), Particle Gibbs with
  Ancestor Sampling + NUTS (Bayesian), PMMH, and a bootstrap particle filter.
  Every backend × method combination either works and is tested, or fails loudly
  through a capability check that names the limitation — never a silent wrong
  answer.
- **Exact gradients, no autodiff tape.** The compiler differentiates rate
  expressions symbolically (source-to-source) and emits the gradient as ordinary
  IR. NUTS gets exact ∇ log ℒ from one expression-tree walk per step — no finite
  differences, no runtime autodiff.
- **Reproducible by construction.** Runs are content-addressed by their inputs;
  paired scenarios share random numbers so a counterfactual differs only where
  the intervention does. Fits ship convergence gates between stages and a
  Richardson dt-convergence audit, so "it ran" and "it converged" are different
  claims.
- **One readable IR, two languages.** An OCaml frontend expands the DSL
  (stratification, contact matrices, Erlang stages, forcing, interventions) into
  a flat JSON intermediate representation; a Rust backend simulates and fits it.
  A version envelope makes a frontend/backend mismatch a hard error, not a
  wrong-but-parseable run.

```text
model.camdl ──→ camdlc ──→ model.ir.json
                                │
          ┌──────────┬──────────┼───────────┐
          ▼          ▼          ▼           ▼
    camdl simulate  camdl fit  camdl batch  camdl compare
          │          │          │           │
          ▼          ▼          ▼           ▼
     trajectory   MLE / posterior  sweep   elpd table
```

More, in one table:

| Domain               | What camdl does                                                                                                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Modelling**        | Compartments; stratification (age, space, risk); contact matrices; Erlang staging; forcing functions; scheduled and reactive interventions; events; balance constraints; scenarios     |
| **Simulation**       | Gillespie SSA, chain-binomial (Euler-multinomial), ODE (RK4 / adaptive RK4(5)). Extra-demographic noise via `overdispersed()`; deterministic flows via `deterministic()`               |
| **Inference**        | IF2 (MLE), PGAS + NUTS (Bayesian posterior), PMMH, bootstrap particle filter, 1D/2D profile likelihood. Compiler-emitted analytic gradients                                            |
| **Fitting workflow** | Declarative `fit.toml` (named stages → `camdl fit run`); convergence gates between stages; dt-convergence audit; calendar-dated data; sparse / irregular / missing (`NA`) observations |
| **Diagnostics**      | Particle-filter health (ESS, τ²), prequential scoring (elpd, CRPS, PIT), paired model comparison via `camdl compare`, posterior-predictive checks via `camdl fit predict`              |
| **Reproducibility**  | Content-addressed runs with caching; common-random-number scenario coupling; version-locked offline docs (`camdl docs <topic>`)                                                        |

## Install

### Quick install (Linux / macOS)

For a fresh machine, the `install.sh` script at the repo root installs both
toolchains (OCaml ≥ 5.2 via opam, Rust stable via rustup), fetches OCaml package
dependencies, and runs `make build && make install`:

```bash
./install.sh
```

It's idempotent — safe to re-run — and **never calls `sudo` or a system package
manager**. Everything it installs lands in paths you own: opam and (if the
system CMake is missing or older than 3.13) a portable CMake go under `$PREFIX`
(default `~/.local`); the OCaml switch under `~/.opam`; Rust under `~/.cargo`.
This makes it work on locked-down, no-root machines (HPC login nodes, older
distros). It expects a few base tools (`make`, `git`, `curl`, `tar`) to already
be present; if any are missing it prints the one-time command to install them
and stops, rather than escalating on your behalf. Override the OCaml switch
version with `OCAML_SWITCH_VERSION=5.2.1 ./install.sh`.

For per-branch testing (e.g. installing a feature branch alongside your default
install for comparison), override the install prefix with the `PREFIX` env var:

```bash
PREFIX=$HOME/.local-camdl-feat ./install.sh        # binaries → $PREFIX/bin
PATH=$HOME/.local-camdl-feat/bin:$PATH camdl ...   # use the feature build
camdl ...                                          # back to the default install
```

Default `PREFIX` is `$HOME/.local` (binaries land in `$HOME/.local/bin`).

The script initializes opam with sandboxing enabled (via `bubblewrap` on Linux,
`sandbox-exec` on macOS). If sandboxed init fails — typically because
`bubblewrap` isn't installed or your kernel disallows unprivileged user
namespaces — the script will abort with instructions. You can re-run with
`NO_SANDBOX=1 ./install.sh` to disable sandboxing, but this reduces supply-chain
protection on every package installed via opam in this switch and is recommended
only when installing `bubblewrap` isn't an option.

If you'd rather wire the toolchain by hand, follow the manual steps below.

### Prerequisites (manual install)

camdl has two language runtimes. You need both available before `make build`
will work:

- **OCaml ≥ 5.2 + opam.** macOS: `brew install opam`. Linux: see
  [opam.ocaml.org/doc/Install.html](https://opam.ocaml.org/doc/Install.html).
  Then create a switch:

  ```bash
  opam init -y                    # first-time only
  opam switch create 5.2.0        # match what CI uses
  eval $(opam env)
  ```

- **Rust stable** via rustup ([rustup.rs](https://rustup.rs/)):
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`, then
  `rustup default stable`.

- **Make + git + python3** (only needed for `make update-golden` and the
  integration test driver — usually already installed).

Once after cloning, install the OCaml package dependencies declared in
`ocaml/*.opam`:

```bash
cd ocaml
opam install . --deps-only --with-test --yes
cd ..
```

This fetches `dune`, `menhir`, `yojson`, `fmt`, `alcotest`, and the qcheck
stack. Skipping it produces errors like `Library "yojson" not found` or
`Program menhir not found in the tree or in PATH` — those mean the opam install
step hasn't run, not that the build is broken.

### Build

```bash
make build       # builds both OCaml and Rust
make install     # copies camdl + camdlc to ~/.local/bin
make test        # OCaml + Rust + integration (~800 tests in this repo)
```

`make install` is required after every rebuild — `camdl` checks the on-PATH
`camdlc` hash matches its own and refuses to run on a mismatch. The post-install
message warns if another `camdl` (e.g. a leftover `cargo install`) is shadowing
on PATH.

Make sure `~/.local/bin` is on your PATH; on most shells that means adding
`export PATH="$HOME/.local/bin:$PATH"` to `~/.zshrc` or `~/.bashrc`.

## Learn more

Every camdl binary ships its own version-matched guides offline — run
`camdl docs` for the topic list (`workflow`, `fit-toml`, `concepts`,
`diagnosing-fits`, `language`, …). The longer-form references:

| Document                                                       | Contents                                         |
| -------------------------------------------------------------- | ------------------------------------------------ |
| [`docs/intro.md`](docs/intro.md)                               | DSL tutorial                                     |
| [`docs/camdl-language-spec.md`](docs/camdl-language-spec.md)   | Full DSL reference                               |
| [`docs/user-features.md`](docs/user-features.md)               | Feature catalog with pomp comparison             |
| [`docs/examples.md`](docs/examples.md)                         | Catalogue of every model shipped with camdl      |
| [`docs/inference.md`](docs/inference.md)                       | Inference guide (PF, IF2, PGAS, NUTS)            |
| [`docs/camdl-alternatives.md`](docs/camdl-alternatives.md)     | How camdl compares to Stan, pomp and others      |
| [`docs/commands.md`](docs/commands.md)                         | Command taxonomy and workflows                   |
| [`docs/runtimes.md`](docs/runtimes.md)                         | Simulation backend details                       |
| [`docs/camdl-data-spec.md`](docs/camdl-data-spec.md)           | IR schema and data model                         |
| [`docs/camdl-inference-spec.md`](docs/camdl-inference-spec.md) | Fitting workflow specification                   |
| [`docs/camdl-run-spec.md`](docs/camdl-run-spec.md)             | Run-system / batch / CAS specification           |
| [`AGENTS.md`](AGENTS.md)                                       | Briefing for AI coding agents working with camdl |

**Using camdl from a downstream project with an AI coding agent?** Start with
[`AGENTS.md`](AGENTS.md). It covers the canonical workflow, error / diagnostic
interpretation, when to stop and ask the human, and a shallow-clone recipe for
pinning the docs locally so the agent can read the language spec offline at
version-matched cost (~5 MB):

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/vsbuffalo/camdl .camdl-source
cd .camdl-source && git sparse-checkout set docs ocaml/golden && cd ..
```

Add `.camdl-source/` to your project's `.gitignore`; sync with
`git -C .camdl-source pull`.

## Status

camdl is **alpha**: the public surface (DSL, CLI, output formats) is documented
and usable for real fits, but breaking changes are still expected before 1.0.
See [`VERSIONING.md`](VERSIONING.md) for what the version number promises and
[`docs/language-changes.md`](docs/language-changes.md) for the migration log.

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
  crates/sim/          Simulation backends (Gillespie, chain-binomial,
                       ODE) + propensity evaluator + inference
                       (PF, IF2, PGAS+NUTS, PMMH, obs_loglik, prequential)
  crates/io/           TSV read/write
  crates/cli/          camdl: simulate, batch, fit, pfilter, profile,
                       survey, data, lineage, list, show, cat, compare,
                       label, check, inspect, docs, mre, dev (compile,
                       doctest, eval, reindex)
ir/
  VERSION              Canonical IR schema version. Single source of truth:
                       Rust reads it via include_str! at compile time; OCaml
                       reads it via a Makefile-generated constant module. Bump
                       this and `make build` to break the IR contract.
  schema.json          IR schema reference for the inner Model (the on-wire
                       shape is wrapped in the envelope
                       { ir_version, validated_by, model } defined by
                       rust/crates/ir/src/envelope.rs).
  golden/              Canonical IR files (Rust test surface, envelope-wrapped)
docs/                  Specifications and guides
benches/              Criterion benchmarks + performance lab notebook
```

## Architecture

The **IR is fully expanded**: the OCaml compiler performs all stratification,
coupling expansion, and let-binding inlining. The Rust backend sees only flat
compartments, transitions, and expression ASTs.

The **expression language** is pure, total, and first-order
(`Const | Param | Pop | PopSum | Time | Dt | BinOp | UnOp | Cond | TimeFunc | TableLookup | Projected | Reduce | BindingRef | …`),
evaluated in bounded time at each simulation step. The same property that makes
dimension-checking tractable makes source-to-source autodiff a compact OCaml
pattern match — exact ∇ log ℒ at one expression-tree walk per step, no autodiff
tape.

The **IR contract** is enforced via a version envelope
(`{ ir_version, validated_by, model }`) — Rust's deserializer rejects mismatched
schemas at the boundary, so OCaml/Rust drift fails CI rather than producing
wrong-but-parseable simulations.

**Common random numbers**: the same seed yields an identical trajectory, so
paired scenarios differ only where the intervention does — the basis for
counterfactual comparisons.

**Strict-mode runtime.** Rate-evaluation degeneracies (division by zero, NaN/Inf
from `Pow`, sqrt of a negative, binomial overshoot) produce typed errors by
default (`SimError::NumericalCollapse`, `NegativeCount`). Inference layers catch
per-particle-recoverable errors and convert them to a −Inf log-likelihood for
the offending particle (resampling kills it, the chain continues); forward
simulation halts. Pass `--allow-degenerate-rates` to restore the legacy
silent-zero for the rare legitimate case.

## License

Apache 2.0 — see [LICENSE](LICENSE). Developed at the
