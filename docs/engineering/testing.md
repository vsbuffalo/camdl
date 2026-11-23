# Testing camdl

camdl's tests check the OCaml compiler, the Rust runtime, the two working
together, and camdl's results against other software, and this page explains
what each group of tests guarantees, how the full suite runs, and how to add a
test. It is written for anyone changing camdl, human or agent. The full suite is
`make test`; the Makefile and the agent instructions in `CLAUDE.md` call it the
gate, because a change lands on `main` only after it passes.

## Commands

```bash
make test-fast        # the whole Rust workspace through cargo test; for iterating
make test             # the full suite, required before a change lands
make test-ocaml       # layers 1 and 2: the compiler, dimcheck and the IR round-trip
make test-rust        # the Rust workspace except the sim crate
make test-inference   # the sim crate: the simulation engine and the inference stack
make test-integration # layer 7: compiled models run by the runtime
make gate-timing      # make test, timed step by step

cd rust && cargo test -p sim --test erlang_distribution            # one test file
cd rust && cargo test -p sim --test pmmh some_test -- --nocapture  # one test, with its output
cd rust && cargo test -p sim -- --ignored                          # layer 6, skipped by default
cd ocaml && dune runtest test/test_compiler.exe --force            # one OCaml suite
```

While you change one language, run its target; run `make test` before the change
lands. Run single Rust tests without `--release`. The test profile already
compiles `sim`, `numerics`, `score` and every third-party crate at opt-level 2,
as `make test` does, and `--release` would build a second copy of the whole
workspace.

sccache is a compiler cache: it wraps `rustc`, stores each compiled crate under
a hash of its inputs, and returns the stored result when the same crate is
compiled again. It is optional (`brew install sccache`); the Makefile uses it as
`RUSTC_WRAPPER` when it is on `PATH`. It caches compiles that are not
incremental, which are the third-party crates and the release build. cargo
compiles the workspace's own crates incrementally in test builds, and sccache
does not cache those.

## The layers of tests

camdl is two programs. The compiler, `camdlc`, written in OCaml, reads a
`.camdl` model, checks it and writes an intermediate representation (IR), a JSON
description of the model. The runtime, `camdl`, written in Rust, reads that IR
to simulate the model and fit it to data. A defect can therefore sit in the
compiler, in the runtime's algorithms, in the command-line program around them,
or in the agreement between the two halves about what the IR means. And a model
can pass every one of those checks and still compute the wrong thing, if camdl
and its tests share the same mistaken idea of what is correct.

The layers below follow that structure, from the compiler outward. Layers 1 to 7
check that camdl agrees with itself: the compiler with the language
specification, the runtime with its own invariants, the runtime with the
compiler. Layer 8 compares camdl with other software and with closed-form
results.

Each layer ends with the same box. Its time is how long that layer's own tests
run in `make test` on an otherwise idle 8-vCPU machine (camdl-cloud run
`gate-share-2`), counting each test binary once and leaving out compiling, which
the [run time](#run-time) table counts for the whole run; the layers' times add
up to about 630 s of its 1,114 s.

### Layer 1: the compiler and dimension checker

These tests guarantee that the compiler accepts what the language specification
allows, refuses what it forbids with the documented error, and expands a model
into the transitions the specification says it has.

`test_compiler.ml` covers parsing, stratification, scenarios, observations,
interventions, and regressions for particular claims the specification makes.
`test_dimcheck.ml` tests the checker of physical dimensions, which refuses, for
example, adding a rate to a count; qcheck property tests sit beside its fixed
cases. `ocaml/test/errors/` holds one minimal model per error code, and the
`negative_golden` suite compiles each with JSON diagnostics on and checks that
the code appears.

::: {.practice title="Running layer 1" collapse="false"}

|         |                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------ |
| Where   | `ocaml/test/test_compiler.ml`, `ocaml/test/test_dimcheck.ml`, `ocaml/test/errors/`                                 |
| Command | `make test-ocaml`, or `cd ocaml && dune exec test/test_compiler.exe -- test 'table_unit_conversion'` for one group |
| Time    | 1.7 s for layers 1 and 2 together, which one `dune runtest` runs                                                   |
| Runs in | `make test`, `compiler.yml`                                                                                        |

:::

### Layer 2: the IR round-trip

These tests guarantee that every model the compiler writes can be read back
unchanged, so the OCaml and Rust sides keep one definition of the IR's format.

`test_ir_roundtrip.ml` compiles every `.camdl` in `ocaml/golden/` to IR,
serializes it, reads it back and compares the two. A golden model is picked up
once its `.ir.json` exists (`make update-golden`).

::: {.practice title="Running layer 2" collapse="false"}

|         |                                                         |
| ------- | ------------------------------------------------------- |
| Where   | `ocaml/test/test_ir_roundtrip.ml`, over `ocaml/golden/` |
| Command | `make test-ocaml`                                       |
| Time    | Counted with layer 1's: one `dune runtest` runs both    |
| Runs in | `make test`, `compiler.yml`                             |

:::

### Layer 3: Rust unit tests

These tests guarantee that individual functions of the runtime, such as a
random-number stream, a resampling step or a hash, do what their documentation
says, on inputs small enough to reason about.

They sit in `#[cfg(test)] mod tests` blocks inside each crate's `src/` files and
compile with the crate, so a change to a function's signature breaks them at
build time.

::: {.practice title="Running layer 3" collapse="false"}

|         |                                          |
| ------- | ---------------------------------------- |
| Where   | `rust/crates/*/src/`                     |
| Command | `cd rust && cargo test -p <crate> --lib` |
| Time    | 39 s                                     |
| Runs in | `make test`, `ci.yml`, `inference.yml`   |

:::

### Layer 4: Rust integration tests in the same process

These tests guarantee that the runtime's algorithms produce correct results end
to end: that a simulation conserves population, that a seed reproduces a
trajectory exactly, that a particle filter or a sampler targets the distribution
it should.

Each file in a crate's `tests/` directory is compiled into a test binary of its
own that links the crate as a library. There are 187 in `sim`, a few in `ir`,
`declared`, `diff`, `score` and `typedoc`, and two at the workspace root in
`rust/tests/`. In `sim`, for example:

| File                                                      | What it checks                                                                                    |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `gillespie_determinism.rs`                                | the same seed gives a byte-identical trajectory                                                   |
| `gillespie_invariants.rs`, `chain_binomial_invariants.rs` | mass conservation and the other invariants of each stochastic backend                             |
| `ode.rs`                                                  | the RK4 backend                                                                                   |
| `particle_filter.rs`, `if2.rs`, `pmmh.rs`                 | the bootstrap filter's log-likelihood, IF2 and PMMH                                               |
| `pgas_exact_posterior.rs`                                 | PGAS's parameter marginal against an exactly enumerated posterior                                 |
| `cubic_spline.rs`, `interpolation.rs`                     | interpolation against SciPy and NumPy reference values                                            |
| `smoke_all_golden.rs`                                     | every golden model simulates under every backend; catches a changed interface, not wrong dynamics |

::: {.practice title="Running layer 4" collapse="false"}

|         |                                                                                                                 |
| ------- | --------------------------------------------------------------------------------------------------------------- |
| Where   | `rust/crates/*/tests/*.rs`, `rust/tests/`                                                                       |
| Command | `make test-inference` for `sim`, `make test-rust` for the rest, or `cd rust && cargo test -p sim --test <file>` |
| Time    | 112 s                                                                                                           |
| Runs in | `make test`, `inference.yml`, `ci.yml`                                                                          |

:::

### Layer 5: the command-line program

These tests guarantee that the `camdl` command behaves as a user sees it:
arguments, files read and written, the results store, messages and exit codes.

Most of the 262 test files in `rust/crates/cli/tests/` run the `camdl` binary in
a temporary directory and check what it writes. They use either the release
binary `rust/target/release/camdl`, which `make test-rust` builds first, or the
binary cargo builds for the test (`CARGO_BIN_EXE_camdl`). A test of the release
binary fails, with a message naming `make build-rust`, when that binary is
missing.

A test that starts `camdl` also needs `camdlc`, and `camdl` refuses a `camdlc`
built from a different git commit than its own. That check protects a user from
a mismatched pair. In development the `camdlc` on `PATH` (usually the shared
`~/.local/bin/camdlc`) was often installed by another checkout at another
commit, and the test then aborts with "camdlc version mismatch". The test
targets put the freshly built `camdlc` first on `PATH` and set
`CAMDL_SKIP_VERSION_CHECK=1`, so `camdl` uses the compiler under test; a
mismatch from a bare `cargo test` is the environment, not the change, and goes
away when the test runs through `make`.

::: {.practice title="Working around the camdlc version check outside make"}

For a `camdl` run by hand, set `CAMDL_SKIP_VERSION_CHECK=1`, or
`CAMDLC=ocaml/_build/default/bin/camdlc.exe`. Do not run `make install` or
`make install-camdlc` to clear a mismatch: they replace the shared
`~/.local/bin/camdlc` that other checkouts use, and a `camdlc-<hash>` beside the
binary from `make dev-camdlc` can shadow the stand-in `camdlc` programs that the
`compile_once` and `ir_cache` tests put on `PATH`. The test targets put the
compiler first on `PATH` rather than setting `CAMDLC` for the same reason:
`CAMDLC` takes priority over `PATH` in `find_camdlc` and would override those
stand-ins. With the check skipped, `camdl` checks the IR instead, and refuses a
document whose `ir_version` differs from `ir/VERSION`, naming both, without
writing it to the IR cache in `~/.cache/camdl/ir`.

:::

::: {.practice title="Running layer 5" collapse="false"}

|         |                                                                                           |
| ------- | ----------------------------------------------------------------------------------------- |
| Where   | `rust/crates/cli/tests/`                                                                  |
| Command | `make test-rust`, or `cd rust && cargo test -p cli --test <file>` after `make build-rust` |
| Time    | 370 s, the largest of the layers                                                          |
| Runs in | `make test`, `ci.yml`                                                                     |

:::

### Layer 6: Monte Carlo tests of distributions

These tests guarantee that the stochastic simulators sample the distributions
the model describes, not only that they run and conserve population. A simulator
can pass every test of layer 4 and still draw waiting times from the wrong
distribution; only a comparison over many seeds shows it.

Each test runs thousands of seeds and compares a summary of the samples with its
expected value, within a tolerance of three Monte Carlo standard errors. The
tolerance comes from that standard error, never from what passes today,
otherwise a drift inside the tolerance cannot be seen. Each test also checks
that the result is distinguishable from a degenerate one, since a different
distribution with a similar mean can pass a check on the mean alone. The pattern
is in `docs/dev/runtime-simulation-tests.md`. Because they are slow, these tests
are marked `#[ignore]` and are in neither `make test` nor CI.

::: {.practice title="Running layer 6" collapse="false"}

|         |                                                                                                                                                                                                                   |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Where   | `rust/crates/sim/tests/statistical_distribution.rs`, `erlang_distribution.rs` and other `#[ignore]` tests                                                                                                         |
| Command | `cd rust && cargo test -p sim -- --ignored`                                                                                                                                                                       |
| Time    | 3–30 s per test                                                                                                                                                                                                   |
| Runs in | Nothing automatically; run them before a release, and after a change to `sim/src/gillespie.rs`, `chain_binomial.rs`, `propensity.rs`, anything under `sim/src/inference/`, or how `expander.ml` emits transitions |

:::

### Layer 7: the compiler and runtime together

These tests guarantee that what the compiler writes, the runtime reads and runs:
that the two agree on the IR in practice, not only in each one's own tests.

`tests/test_ocaml_to_rust.sh` compiles every golden model with `camdlc` and
simulates its IR on each backend. It then runs eight golden models as
`simulate --sweep … --replicates` grids into a fresh results store, checks the
number of results through `camdl list`, and runs each grid again to check that
every cell is read back from the store. It catches IR the runtime cannot read, a
shape the compiler writes and `simulate` refuses, and a renamed command. The
models are `ocaml/golden/<name>.camdl` with `<name>.params.toml`; the grid flags
are in the script.

::: {.practice title="Running layer 7" collapse="false"}

|         |                                                     |
| ------- | --------------------------------------------------- |
| Where   | `tests/test_ocaml_to_rust.sh`, over `ocaml/golden/` |
| Command | `make test-integration`                             |
| Time    | 27 s                                                |
| Runs in | `make test`, the pre-push hook, `ci.yml`            |

:::

### Layer 8: comparison with other software

These tests guarantee that camdl's results agree with independent
implementations of the same models, which catches a defect the other layers
cannot: one that is dimensionally valid and consistent with every other test,
such as a rate converted to the wrong time unit, because camdl and its own tests
share the mistake.

Each case runs a model in camdl and compares a summary with the same summary
from a reference, such as [pomp](https://doi.org/10.18637/jss.v069.i12) (King,
Nguyen and Ionides 2016), an R package for inference on partially observed
Markov processes, or [NumPyro](https://arxiv.org/abs/1912.11554) (Phan, Pradhan
and Jankowiak 2019), or a closed-form result. A case fails when the two differ
by more than a tolerance set for that case. In `make test` the harness runs
camdl alone and compares it with reference results committed to the repository;
no R, Python or Stan is needed. Regenerating a reference runs the reference
software itself.

| Case                           | What it compares                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sir_analytical`               | a closed SIR model at R0 = 3 with the final size given by [Kermack and McKendrick (1927)](https://doi.org/10.1098/rspa.1927.0118)                       |
| `boarding_school_sir`          | pomp's SIR tutorial model, the boarding-school influenza outbreak                                                                                       |
| `he2010_forward`               | simulations of the London measles model of [He, Ionides and King (2010)](https://doi.org/10.1098/rsif.2009.0151) with pomp's, at the published estimate |
| `he2010_pfilter_loglik`        | the particle filter's log-likelihood for that model with pomp's `pfilter()`, at matched parameters, particle count and time grid                        |
| `he2010_pfilter_loglik_sparse` | the same, with sparse observations                                                                                                                      |

A case is a directory `tests/external/cases/<name>/` holding `case.toml` (what
camdl runs), `expected.toml` (the tolerances, each with the Monte Carlo power
that justifies it), `reference/` (the reference driver and its pinned
dependencies) and `fixtures/` (the cached reference summary and a manifest). The
manifest records hashes of the reference directory, the case files and the
harness version, and a mismatch fails the test with the command that regenerates
the reference, so a reference cannot drift out of date unnoticed. Adding a case
is described in `tests/external/README.md`; the design is
`docs/dev/proposals/archive/pre-alpha/2026-04-23-external-validation-harness.md`.

::: {.practice title="Running layer 8" collapse="false"}

|         |                                                                                                                                                                                                                                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Where   | `tests/external/cases/`, through `rust/tests/external_validation.rs`                                                                                                                                                                                                                                                |
| Command | `cd rust && cargo test --test external_validation -- --nocapture` for each case's result; `cargo build -p external-harness`, then `./target/debug/external-harness run tests/external/cases/<case>` for one case, or `CAMDL_REGEN_EXTERNAL=1 ./target/debug/external-harness run-all` to regenerate every reference |
| Time    | 82 s, mostly the two particle-filter cases                                                                                                                                                                                                                                                                          |
| Runs in | `make test` (as part of layer 5's run), `ci.yml`                                                                                                                                                                                                                                                                    |

:::

### When layers disagree

The layer that fails, given the ones that pass, says where to look. If layer 4
passes and layer 5 fails, the defect is in the command-line program: argument
parsing, path resolution, or how it applies parameters and scenarios to the
model. If layer 5 passes and layer 7 fails, the compiler and the runtime
disagree about the IR. If layers 1 to 4 pass and layer 6 fails, the compiler
writes IR that looks right and simulates wrongly; a defect in converting table
units was of that kind, with the compiler's test passing and no runtime test of
the dynamics.

## How `make test` runs

`make test` runs these steps one after another. Each is also a target of its
own.

| Step                       | Target                              | What it checks                                                                 |
| -------------------------- | ----------------------------------- | ------------------------------------------------------------------------------ |
| Lint                       | `lint`                              | `cargo clippy --workspace --all-targets -D warnings`                           |
| Compiler                   | `test-ocaml`, the `check-*-golden`s | layers 1 and 2, and that the committed golden IR files still compile unchanged |
| Release build              | `build-rust`                        | the `camdl` binary that layer 5 runs                                           |
| The rest of the workspace  | `test-rust`                         | `cargo test --workspace --exclude sim`: layers 3 to 5 and 8 outside `sim`      |
| Simulation and inference   | `test-inference`                    | `cargo test -p sim`: layers 3 and 4 in `sim`                                   |
| Compiler and runtime       | `test-integration`                  | layer 7                                                                        |
| Documentation and the rest | `test-docs`, `test-cli-docs`, …     | the specification's examples, the CLI pages, the install script, these scripts |

`cargo test` runs a crate's test binaries one at a time; within a binary, tests
run in parallel on `RUST_TEST_THREADS` threads. A binary holding one long test
therefore runs on one thread, however many cores the machine has.

The test profile compiles `sim`, `numerics`, `score` and every third-party crate
at opt-level 2, with debug assertions and overflow checks on (`rust/Cargo.toml`,
`[profile.dev.package.*]`). The particle-filter, CSMC and MCMC tests run
thousands of iterations; unoptimized they took 955 s of a 2,098 s run of
`make test`, and optimized they take about 100 s, with every outcome unchanged
([the measurement](../dev/notes/2026-11-21-optimized-test-profile.md)).

`make test-fast` runs the whole Rust workspace and skips `dune runtest`, the
golden checks, layer 7 and the documentation checks. CI runs every step of
`make test`, so what the fast run skips is still checked before a change merges.

### Sharing a machine

Several checkouts on one machine, such as agents working in separate worktrees,
can each start `make test`. Seven at once once drove a 16-core machine to a load
of about 145, and every run slowed. Two mechanisms divide the machine: a queue
limits how many runs go at once, and each run sizes its threads to its share.

**The queue.** `scripts/test-slot.sh` is a counting semaphore. `make test`,
`make test-rust`, `make test-inference`, `make test-integration` and the
pre-push hook each hold one of `CAMDL_TEST_SLOTS` slots while they run, and wait
while all are taken, printing `waiting for a test slot (3 in use; …)` once.
`make test-fast` and single `cargo test` runs are not queued.

- A slot is an exclusive `flock(2)` lock (`lockf` on macOS, `flock` on Linux) on
  one of the files `slot.0` … `slot.N-1`, so the kernel releases it when the
  holder exits, however it exits, `kill -9` included.
- A run takes one slot for all its steps: the queued target runs `make <target>`
  again inside its slot with `CAMDL_TEST_SLOT_HELD=1` exported, and under that
  marker every queued target runs straight through.
- Waiting runs retry every two seconds, in no particular order.
- The slot directory is `/tmp/claude-<uid>/camdl-test-slots/` when that
  directory exists (the Claude Code sessions of one user share it), else
  `${TMPDIR:-/tmp}/camdl-test-slots/`; `CAMDL_TEST_SLOT_DIR` overrides it.
- `CAMDL_TEST_SLOTS=0` turns the queue off. CI sets it in `ci.yml`,
  `inference.yml` and `release.yml`, where one job has the runner to itself.
- `make test-slot-queue` tests the queue: with two slots and three jobs, at most
  two run at once; a holder killed with `SIGKILL` frees its slot; a process the
  command leaves behind does not hold it; the exit status is the command's.

**The share.** Each run uses the machine's cores divided by `CAMDL_TEST_SLOTS`
(all of them with the queue off): that many compile jobs (`CARGO_BUILD_JOBS`),
that many test threads per test binary (`RUST_TEST_THREADS`), and half as many
Rayon threads (`RAYON_NUM_THREADS`) in each test binary and each `camdl` a test
starts. Rayon is the library camdl's parallel loops use. The Rayon count is half
the share because a binary running several tests at once starts one Rayon pool
per test, and at the full share those binaries ran slower from the extra
threads. Tests that compare thread counts build pools of an explicit size.

| Machine                            | `CAMDL_TEST_SLOTS`        | Compile jobs | Test threads | Rayon threads |
| ---------------------------------- | ------------------------- | ------------ | ------------ | ------------- |
| A laptop with one checkout         | unset (1)                 | all cores    | all cores    | half          |
| A 16-core machine shared by agents | 3, set in its environment | 5            | 5            | 2             |
| An 8-vCPU cloud machine            | unset (1)                 | 8            | 8            | 4             |
| CI, one job per runner             | 0 (queue off)             | all cores    | all cores    | half          |

A machine meant to run several at once sets the slot count in the environment of
every shell that starts one, non-interactive shells included; in zsh,
`export CAMDL_TEST_SLOTS=3` goes in `~/.zshenv`. An explicit value of any of the
three variables in the environment overrides its share.
`rust/.cargo/config.toml` caps a plain `cargo test`, outside `make`, at 4 test
threads. The Rayon setting is in the Makefile rather than that file's `[env]`,
which cargo also applies to `cargo run` and `cargo bench`, where it would time
parallel code on fewer threads without saying so.

### Skipping a run that already passed

The pre-push hook runs part of `make test`, so a push straight after a passing
`make test` would repeat work. A passing run therefore records the git tree hash
it tested, the hash of every tracked file's path and content, as a file in
`<git common dir>/camdl-gate-passed/`, shared by every worktree of the
repository, and the hook exits at once when every pushed commit's tree has a
record. A record also covers an amended message or a rebase that leaves the
content unchanged.

- `scripts/gate-stamp.sh` writes a record only when the checkout had no
  uncommitted change, and no untracked file under `rust/`, `ocaml/`, `tests/`,
  `ir/`, `scripts/`, `docs/` or `.githooks/`, from the start of the run to its
  end. Records are removed after 14 days.
- A passed camdl-cloud run of `make test` counts as a record. camdl-cloud is a
  separate tool, used by the maintainers, that runs commands on Google Cloud
  machines. `camdl-cloud test --rev <commit>` clones that commit on a fresh
  machine, refuses to run unless its tree is clean, runs `make test` and records
  the outcome. The hook reads the records with `camdl-cloud runs --json` and
  accepts a pushed commit whose tree is the tree of a run with target `test` and
  outcome `passed`. Without `camdl-cloud` on `PATH`, or with
  `CAMDL_GATE_CLOUD=0`, only local records count.
- `make test-gate-stamp` tests these rules.
- A record covers the repository's content, not state outside it such as the
  installed opam packages; CI runs the full suite on every push and pull request
  that changes code, record or not.

Keying a test result by a hash of exactly what was tested is how
[Bazel](https://bazel.build/reference/command-line-reference#flag--cache_test_results)
caches test results; here the hash is the tree hash git already keeps.

With a cloud pass, a change lands without running the suite on the developer's
machine: commit, `camdl-cloud test --rev HEAD`, wait for `passed`, push. The
cloud machine runs linux/arm64 and CI runs linux/x86-64, so a failure that
occurs only on macOS is found by the next local run rather than before the push.

### The pre-push hook

`.githooks/pre-push`, installed through `core.hooksPath`, runs on every
`git push` unless every pushed tree has a record, inside a test slot:

1. the OCaml build and `dune runtest`;
2. `cargo test --workspace --no-fail-fast`;
3. `cargo clippy --workspace --all-targets -- -D warnings`;
4. `make update-golden`, then a check that `ir/golden/`, `ocaml/golden/`,
   `tests/fixtures/reactive/ir/` and `tests/fixtures/regression/ir/` are
   unchanged;
5. `make test-integration`.

Never skip it. It exists because a commit that compiled its tests without
running them once reached CI and failed there.

### CI

The workflows in `.github/workflows/` run the steps of `make test` in parallel
on every push to `main` and every pull request, each when the files it depends
on change:

| Workflow                                                 | Runs                                                                                                       |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [`ci.yml`](../../.github/workflows/ci.yml)               | clippy, `make test-rust`, the golden check over all four golden directories, `make test-integration`       |
| [`inference.yml`](../../.github/workflows/inference.yml) | `make build-benches` (every benchmark compiles), then `make test-inference`                                |
| [`compiler.yml`](../../.github/workflows/compiler.yml)   | `make test-ocaml`                                                                                          |
| [`doctest.yml`](../../.github/workflows/doctest.yml)     | `make test-docs`: the examples in the specifications, through `camdlc doctest`                             |
| [`cli-docs.yml`](../../.github/workflows/cli-docs.yml)   | `make test-cli-docs`                                                                                       |
| [`data-spec.yml`](../../.github/workflows/data-spec.yml) | `make test-data-spec`: every observation-file example in `docs/camdl-data-spec.md` through the real loader |
| [`dates-doc.yml`](../../.github/workflows/dates-doc.yml) | `make test-dates-doc`: every `simulate` table in `docs/dates.md`, its time and date columns                |
| [`release.yml`](../../.github/workflows/release.yml)     | the release builds for Linux, macOS and Windows                                                            |

`data-spec.yml` and `dates-doc.yml` exist because `ci.yml` does not run on a
change to `docs/` alone; their harnesses are also `cli` tests, which `ci.yml`
runs. Layer 6 is in no workflow.

### Run time

`make test` at commit `95bdde19b`, on an otherwise idle 8-vCPU machine
(`c4a-highcpu-8`, 16 GB, linux/arm64) from cold caches, took 1,149 s
(camdl-cloud run `gate-land-1`). Its steps, from a timed run with the same
settings that took 1,114 s (`gate-share-2`); the steps run one after another, so
each one's share of the run is its time over the total. Repeated runs of one
commit agree to about 20 s.

| Step                                     | Time  | Share of the run |
| ---------------------------------------- | ----- | ---------------- |
| The rest of the workspace: running tests | 505 s | 45 %             |
| Simulation and inference: compiling      | 177 s | 16 %             |
| The rest of the workspace: compiling     | 134 s | 12 %             |
| Simulation and inference: running tests  | 98 s  | 9 %              |
| Release build                            | 97 s  | 9 %              |
| Lint                                     | 56 s  | 5 %              |
| Everything else                          | 45 s  | 4 %              |

While tests run, much of the machine is idle: cargo runs a crate's test binaries
one at a time, and a binary with one long test runs on one thread.

With fixed settings of six compile jobs, four test threads and two Rayon threads
the same commit took 1,433 s (`gate-main-3`), and before the optimized test
profile 2,098 s (`gate-base-1`)
([the measurement](../dev/notes/2026-11-21-gate-share.md)).

Starting a test binary costs nothing measurable on Linux: over 482 binaries, the
time from cargo starting each one to its tests, less the time the binary reports
for its tests, totals 1–2 s. On macOS the first launch of a newly linked binary
waits for the system's security assessment, which measured 838 s over one run on
a loaded 16-core machine. That is why `make test` does not use `cargo-nextest`:
nextest starts many test binaries at once, and that burst of first launches
stalled the assessment on macOS and hung every launch.

`make gate-timing` runs `make test` (or `TARGET=<name>`) with every recipe line
and every line of output timestamped, keeps the log in
`rust/target/gate-timing/`, and prints the wall time and the time spent waiting
for a slot, the longest recipe lines, each cargo invocation split into
compiling, running tests and the remainder, and each test binary's launch
overhead. `scripts/gate_timing_report.py <log>` prints the report again. The
timed run is a real `make test`: it queues for a slot and records a pass.

## Writing tests

Which kind of test a change needs follows from what it changes. A new claim in
the specification needs a test that would fail if the claim stopped holding. A
new error needs a model that raises it. A change to how the simulators draw
needs a statistical check. A new model the other tests should exercise becomes a
golden model. The sections below take each in turn.

### Regression tests for specification claims

The specification makes claims the compiler and runtime must uphold: that a
table's values are converted to the model's time unit, that a scenario's `set`
is applied when the model runs. A claim no test checks can break unnoticed, and
the table-unit conversion once did, silently scaling a model's inputs. So a
defect against a claim is fixed test first. We write the test, run it against
the unfixed code to see it fail for the reason we expect, fix the code, see it
pass, and commit both in one change whose message names the section the test
guards (§X.Y). Seeing the test fail first is what shows it can detect the
defect; a test written after the fix may pass for an unrelated reason.
`rust/crates/cli/tests/scenario_runtime_application.rs` and the
`table_unit_conversion` group in `ocaml/test/test_compiler.ml` are examples.

### Error-code fixtures

Every error the compiler can raise has a code, such as E272, and the language
specification says which construct raises it and why. A code is only as
trustworthy as the evidence that the compiler raises it in the situation the
specification describes, so each code gets a fixture: a model that raises that
error and nothing else. You add `ocaml/test/errors/ennn_<slug>.camdl` for the
code raised by an `emit_error ctx ~code:"ENNN" …` call, and check that
`camdlc check` on it reports the code. The `negative_golden` suite in
`test_compiler.ml` finds the new file without further code.

### Monte Carlo tests of distributions

A statistical test of layer 6 draws many samples from a simulator and compares a
summary with its expected value, within a band of three Monte Carlo standard
errors. In outline:

```rust
#[test]
#[ignore = "statistical test: run with --ignored"]
fn my_distributional_claim() {
    let model = setup_isolated_fixture(...);
    let compiled = CompiledModel::new(model).unwrap();
    let mut samples = Vec::with_capacity(n_seeds);
    for seed in 0..n_seeds {
        let traj = GillespieSim.run(&compiled, &params, seed, &config).unwrap();
        samples.push(extract_summary(&traj));
    }
    let actual = mean(&samples);
    let tol = 3.0 * monte_carlo_se(n_seeds, /* sample variance */);
    assert!((actual - expected_from_reference).abs() < tol, "diagnostic …");
}
```

Choose the number of seeds so the band is narrow enough to detect the error you
are guarding against, and add an assertion that the samples differ from the
degenerate case. `docs/dev/runtime-simulation-tests.md` works through the
pattern.

### Golden models

A golden model is a `.camdl` file whose compiled IR is committed beside it, in
`ocaml/golden/`. The IR round-trip of layer 2, `smoke_all_golden.rs` in layer 4
and the cross-language run of layer 7 all exercise every golden model, so adding
one puts a new kind of model under all three. Write `ocaml/golden/<name>.camdl`,
and `<name>.params.toml` if simulations need parameter values, then run
`make update-golden` to write `<name>.ir.json`. Read that JSON before committing
it: it becomes the reference the other tests compare against, so a value you did
not expect is a compiler defect to investigate, not a file to accept.
`ir/golden/` is a separate, frozen set that `make update-golden` does not
regenerate.

When a compiler change alters golden IR on purpose, run `make update-golden`,
read each changed file's diff to check the values are the intended ones, and
commit the compiler change and the golden files together, staged by name so a
formatter's rewrite of other `*.ir.json` files does not ride along. The changed
golden files are then the commit's subject, which is how a reviewer sees them. A
broad `git add` once committed a formatter's pretty-printed rewrite of 48 golden
files under an unrelated documentation commit; every golden file then differed
from what the compiler writes, and CI did not report it for four days because an
earlier step of the same job was already failing
([the incident](../dev/incidents/2026-06-09-golden-format-reverted-by-autoformat.md)).

## Gotchas

- **`#[ignore]` makes a test opt-in.** Run `-- --ignored` before merging a
  change to the simulation or inference code.
- **Use a deterministic backend for checks of a single value.** A command-line
  test whose assertion is on one number should pass `--backend ode`: a
  stochastic backend adds noise from the seed that can hide an off-by-one or a
  change with no effect.
- **Tests in one binary run at once.** Tests that write to a shared path such as
  `/tmp` can race; write to `tempdir()`. To confirm a suspected race, run the
  binary with `-- --test-threads=1`.
- **Golden files after a schema change.** The pre-push hook regenerates the
  golden files and refuses a push that changes them; commit the regenerated
  files in the same branch.
