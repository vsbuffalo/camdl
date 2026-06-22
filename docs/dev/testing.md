# Testing camdl

Orientation for anyone (human or agent) writing or running tests in this repo.
camdl is dual-language (OCaml compiler + Rust runtime) with cross-language
integration, so "what test should I run?" has several correct answers depending
on what you're changing.

## TL;DR commands

```bash
# Inner loop while iterating — the whole Rust workspace (unit + integration +
# doctests) via `cargo test`. Skips the slow cross-language and doc phases, so it
# is faster than the full gate but NOT authoritative (see "Tiered gate" below).
make test-fast

# Authoritative gate — every phase; mirrors CI. SLOW (cross-language integration
# + external pomp/NumPyro validation). Run before a change lands, or let CI run
# it for you.
make test

# Just one language / layer:
make test-ocaml       # OCaml compiler + dimcheck + IR round-trip
make test-rust        # Rust workspace except sim (cargo test)
make test-inference   # the sim crate (engine + inference stack)
make test-integration # cross-language CLI shell-out (slow)

# Statistical tests (slow; skipped by default):
cd rust && cargo test --release --workspace -- --ignored

# A single Rust test file:
cd rust && cargo test --release -p sim --test erlang_distribution

# A single Rust test, with println! output visible:
cd rust && cargo test --release -p sim --test foo test_name -- --nocapture

# A single OCaml suite (Alcotest):
cd ocaml && dune runtest test/test_compiler.exe --force
```

Setup: optionally install **sccache** (`brew install sccache`) as a compile
cache — the Makefile uses it only when it's on PATH (`RUSTC_WRAPPER`), and it
shares artifacts across git worktrees, which speeds the worktree-parallel
workflow. (cargo-nextest was evaluated as a faster runner but its parallel
test-list enumeration spawns a launch burst that wedges macOS
`syspolicyd`/code-signing, hanging every launch in `dyld`; `cargo test` runs
binaries sequentially and is unaffected. sccache stays — it wraps rustc, no
process burst.)

If you're only changing one language, run just that language's layer during
iteration; run the full `make test` (or rely on CI) before a change lands.

## Architecture

Tests are organised by **layer**, not by file type. Each layer answers a
different question about the system; don't substitute one for another.

```
┌───────────────────────────────────────────────────────────────┐
│  Layer                     Where                    When      │
├───────────────────────────────────────────────────────────────┤
│  L1  Parser + type check   ocaml/test/test_compiler.ml        │
│                            ocaml/test/test_dimcheck.ml        │
│                            + ocaml/test/errors/*.camdl        │
│                                                               │
│  L2  IR round-trip         ocaml/test/test_ir_roundtrip.ml    │
│                                                               │
│  L3  Rust unit tests       rust/crates/*/src/**  #[test] mods │
│                                                               │
│  L4  Rust integration      rust/crates/*/tests/*.rs           │
│      (fast)                                                   │
│                                                               │
│  L5  Rust integration      rust/crates/cli/tests/*.rs         │
│      (CLI shell-out)         — shell out to built binary      │
│                                                               │
│  L6  Statistical /         rust/crates/sim/tests/             │
│      distribution            erlang_distribution.rs,          │
│                              statistical_distribution.rs      │
│                              — #[ignore] by default           │
│                                                               │
│  L7  Cross-language        tests/test_ocaml_to_rust.sh        │
│      integration             — compile .camdl → simulate      │
│                                                               │
│  L8  Book build (prose)    ../camdl-book  (external repo)     │
│                                                               │
│  L9  External validation   tests/external/cases/              │
│      (vs pomp, analytical)   — external-harness binary        │
└───────────────────────────────────────────────────────────────┘
```

**Self-consistency vs external validation.** L1–L8 all answer "does camdl agree
with itself?" — golden IR matches source, Rust matches OCaml, synthetic fits
recover truth, etc. L9 answers "does camdl agree with the outside world?" by
comparing against pomp, NumPyro, or closed-form solutions at the same
parameters. Both classes are necessary: self-consistency catches regressions
fast and cheap; external validation catches the class of bug where camdl is
internally consistent but scientifically wrong (GH #11 being the case in point).

### L1 — OCaml compiler + dimcheck

Runs via `dune runtest` in `ocaml/`. Fast (< 1 s). Three suites:

- **test_compiler.ml** (~120 tests): parsing, stratification expansion, scenario
  resolution, observations, interventions, spec-claim regression tests
  (`spec_claims_v1`, `table_unit_conversion`).
- **test_dimcheck.ml** (~73 tests): dimensional analysis checker. Uses qcheck
  for property-based tests alongside the fixture tests.
- **errors/*.camdl + test_compiler's negative_golden suite**: one
  minimum-reproducer `.camdl` per error code. The test compiles each with
  `Diagnostics.json_errors_mode` on and asserts the emitted error code appears
  in the payload. Pattern described in the 2026-04-21 spec-claims audit as the
  right way to grow error-code coverage.

**Running a subset:**

```bash
cd ocaml
dune runtest test/test_compiler.exe --force
dune exec test/test_compiler.exe -- test 'table_unit_conversion'
```

### L2 — IR round-trip

`test_ir_roundtrip.ml`: every `.camdl` in `ocaml/golden/` compiles to IR,
serialises, deserialises, and compares structurally. Catches schema drift.
Automatically exercises every golden fixture — if you add a `.camdl` to
`ocaml/golden/`, regenerate its `.ir.json` via `make update-golden` and the
round-trip test picks it up.

### L3 — Rust unit tests

`#[cfg(test)] mod tests { … }` inside each `src/` file. Fast;
compilation-coupled so they catch API mismatches at build time. Currently
concentrated in: `compiled_model.rs`, `hashing.rs`, `inference/prequential.rs`,
`inference/resampling.rs`, `rng.rs`.

### L4 — Rust integration (fast, in-process)

`rust/crates/sim/tests/*.rs`. Integration tests that import `sim` as a library
(not shell out). Fast — each file compiles once and runs quickly. Highlights:

| File                                               | What it tests                                                                                                                      |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `cubic_spline.rs`                                  | `CubicSpline` vs `scipy.interpolate.CubicSpline(bc_type='natural')` — 12 reference points                                          |
| `interpolation.rs`                                 | Linear + constant interp vs `np.interp` + `interp1d(kind="previous")`                                                              |
| `gillespie_determinism.rs`                         | Same seed → byte-identical trajectory (CRN)                                                                                        |
| `gillespie_invariants.rs`                          | Mass conservation, no-event dynamics, etc.                                                                                         |
| `chain_binomial_invariants.rs`                     | Same invariants for the chain-binomial backend                                                                                     |
| `ode.rs`                                           | RK4 backend correctness                                                                                                            |
| `particle_filter.rs`                               | Bootstrap filter log-likelihood consistency                                                                                        |
| `if2.rs`                                           | IF2 convergence sanity                                                                                                             |
| `pmmh.rs` / `pgas_resume.rs` / `pgas_tempering.rs` | PMMH / PGAS                                                                                                                        |
| `obs_level_params.rs`                              | Observation-model parameter plumbing                                                                                               |
| `interventions.rs`                                 | Intervention timing + state effects                                                                                                |
| `periodic_forcing.rs`                              | Periodic bin lookup                                                                                                                |
| `expr_eval.rs`                                     | Pure expression evaluator                                                                                                          |
| `smoke_all_golden.rs`                              | Every `.ir.json` in `ocaml/golden/` compiles + simulates under every backend — catches crate-level API drift but NOT dynamics bugs |

### L5 — Rust integration (CLI shell-out)

`rust/crates/cli/tests/*.rs`. Each test spawns the built `target/release/camdl`
binary against a `tempdir()` workspace. Slow (each invocation pays the full
binary startup) but tests the end-user surface.

| File                              | What it tests                                                                      |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| `backend_provenance.rs`           | Simulate auto-matches fit's backend; warns on mismatch                             |
| `cas_integration.rs`              | `camdl simulate --cas` + `camdl list/show/cat`                                     |
| `intervention_event_defaults.rs`  | Spec §14.4: events on, interventions off                                           |
| `pfilter_trajectories.rs`         | `pfilter --save-paths N` writes the right shape                                    |
| `scenario_runtime_application.rs` | Spec §17.1: `set`/`scale` actually applied at runtime (closed audit gap P1.1/P1.2) |
| `synthetic_fit_grid.rs`           | `fit run` replicate-grid end-to-end                                                |

**Gotcha: the binary must be built first.** These tests
`skip_if_missing_binary()` when `target/release/camdl` doesn't exist. They
silently skip, not fail. Always run `cargo build --release -p cli` before a full
integration pass, or use `make test-integration` which builds first.

**Gotcha: camdlc version check.** `camdl` (runtime) refuses to run a `camdlc`
(compiler) whose git hash differs from its own — an end-user safeguard against a
drifted runtime/compiler pair. In development it false-reds: the `camdlc` on
PATH (usually the shared `~/.local/bin/camdlc`) goes stale the moment a parallel
checkout runs `make install` at another commit, and the cargo acceptance tests
(`compile_once`, `acceptance_batch_*`) then abort with _"camdlc version
mismatch"_.

**`make test` handles this for you.** `test-rust` prepends the freshly-built
`camdlc` to PATH and skips the handshake, so `camdl` resolves the compiler under
test regardless of `~/.local/bin` state — a plain `make test` is the gate; you
do not need to install or sync `camdlc` first. A "camdlc version mismatch" from
a bare `cargo test` (run directly, not via `make`) is almost always this
environmental issue, **not** your change — re-run under `make test`, or point
`CAMDLC` at the fresh camdlc, before suspecting your diff.

For ad-hoc work outside `make test`:

- Running `camdl <model>.camdl` directly: set `CAMDL_SKIP_VERSION_CHECK=1`, or
  `CAMDLC=ocaml/_build/default/bin/camdlc.exe` to use the fresh one.
- `make dev-camdlc` / `make install-camdlc` resync a matched `camdlc` for
  interactive use. Do **not** run them merely to green a `make test`: it is
  unnecessary now; `make install*` clobbers the shared `~/.local/bin` that
  parallel checkouts depend on; and a binary-adjacent `camdlc-<hash>` from
  `dev-camdlc` can shadow the PATH shims `compile_once`/`ir_cache` inject.

Mechanism note: the harness pins the compiler via **PATH-prepend**, not the
`CAMDLC` env var. `compile_once`/`ir_cache` inject their own counting `camdlc`
shim through PATH; a `CAMDLC` env (`find_camdlc` priority 2) would override
those shims and break them, whereas a PATH-prepended fresh camdlc (priority 3)
coexists — a test that prepends its own shim still wins.

**Note on the binary name.** The binary is `target/release/camdl`. It was called
`camdl-sim` before the 2026-04-20 clap 4 migration; if you have a `camdl-sim`
symlink in `target/release/` from an older checkout, it's harmless but no longer
referenced by any test.

### L6 — Statistical / distribution tests

`rust/crates/sim/tests/statistical_distribution.rs`,
`rust/crates/sim/tests/erlang_distribution.rs`. Marked **`#[ignore]`** because
each test runs thousands of Gillespie seeds and takes ~3-30 s.

**Run them periodically, not every commit:**

```bash
cd rust && cargo test --release -p sim -- --ignored
```

**When to run:**

- Before a release.
- After touching `sim/src/gillespie.rs`, `chain_binomial.rs`, `propensity.rs`,
  or anything in `inference/`.
- After a compiler change to `expander.ml` that affects transition emission
  (e.g., the `consecutive()` staging or stoichiometry).
- Nightly in CI (not configured yet — see audit follow-ups).

**Pattern and tolerance design** → `docs/dev/runtime-simulation-tests.md`. Key
point: tolerance should be computed from Monte-Carlo SE with a 3σ band, not
tuned to pass today. A drift-within-tolerance regression won't be visible
otherwise.

### L7 — Cross-language integration

`tests/test_ocaml_to_rust.sh`. Compiles every `.camdl` fixture with `camdlc`,
feeds the IR to `camdl batch run`, checks exit status. Invoked via
`make test-integration`. Catches:

- OCaml emits IR that Rust can't deserialise (schema drift).
- Rust `batch run` rejects a shape the OCaml compiler happily emits.
- CLI surface renames (the `simulate batch` → `batch run` rename in 2026-04-20
  broke this script until we updated the invocation).

Fixtures live in `tests/fixtures/exp_*.toml`. Each is a batch sweep config
pointing at an `ocaml/golden/*.camdl`.

### L8 — Book build (external repo)

The book lives in `../camdl-book` now (Quarto-based, rendered separately). Its
cells execute camdl commands during render and thereby integration-test the CLI
surface — cell failures there often catch upstream CLI changes that the
Rust/OCaml unit tests don't exercise. See `../camdl-book/CLAUDE.md` for the
render workflow; its own CI runs the book build on PRs touching that repo.

When making CLI-affecting changes in this repo, also render the book locally
(`cd ../camdl-book && uv run quarto render`) before pushing, or expect the
book's CI to catch it.

### L9 — External validation (against pomp, NumPyro, closed-form, …)

The other test layers are **self-consistency** checks: they verify camdl does
what camdl's authors think camdl does. L9 compares camdl's output to an
**external reference** (pomp, NumPyro, Stan, an analytical solution) and fails
when the two disagree beyond a per-case tolerance. Motivated by GH #11
(2026-04-23): the iota miscast and forcing-rescale double-conversion bugs were
dimensionally valid, passed every internal test, and were only detectable
against pomp. L9 exists to close that class of gap.

```bash
# Fast path via cargo test — every `make test-rust` runs this; no
# external tooling needed (cached reference fixtures only).
cargo test --test external_validation --manifest-path rust/Cargo.toml

# Same, with per-case pass/fail visible:
cargo test --test external_validation --manifest-path rust/Cargo.toml -- --nocapture

# Direct harness invocation (equivalent; better for iterative debugging):
cd rust && cargo build -p external-harness
./target/debug/external-harness run-all                        # all cases
./target/debug/external-harness run tests/external/cases/<case>  # single case

# Regen path — re-runs the reference tool (R + pomp, Python + NumPyro, …)
# and refreshes the cached fixture. Requires the reference runtime.
CAMDL_REGEN_EXTERNAL=1 ./target/debug/external-harness run-all
# or equivalently:
./target/debug/external-harness regen tests/external/cases/<case>
```

Cases live in `tests/external/cases/<name>/` with a `case.toml` (what to run),
an `expected.toml` (tolerances + required Monte Carlo power rationale per
check), a `reference/` directory (the external driver + pinned dependencies),
and a `fixtures/` directory (cached summary + MANIFEST.toml). Staleness is
detected via three sha256 hashes (reference directory, case files, harness
version) — any mismatch fails with a regen instruction; there is no silent
drift.

Current cases:

- `sir_analytical` — bare SIR at R0=3 vs Kermack–McKendrick final- size. Zero
  external runtime; the harness's own dogfood.
- `he2010_forward` — He et al. 2010 London measles vs pomp at the published MLE.
  Regression lock for GH #11.
- `boarding_school_sir` — pomp's canonical bare SIR tutorial (Anderson & May
  1991 boarding-school flu).
- `he2010_pfilter_loglik` — particle-filter log-lik at matched (parameters,
  particles, time grid) vs pomp's pfilter. Validates the observation-likelihood
  path and resampling algorithm, not just the simulator.

L9's fast path runs as part of `make test-rust` (and therefore the pre-push hook
and CI). The `external_validation` test at `rust/tests/external_validation.rs`
shells out to the `external-harness` binary; cargo-test reports it as a single
`ok` line, and the per-case breakdown appears inline in test output. Failure
messages include tolerance diffs and a hint to re-run with `--nocapture` for
full detail.

On fixture staleness (edited `reference/` or `model.camdl` / `params.toml`
without regenerating): the test fails with a clear STALE message and the exact
`CAMDL_REGEN_EXTERNAL=1` command to regenerate.

**Design reference:**
`docs/dev/proposals/2026-04-23-external-validation-harness.md`.

**Operator reference (how to run, how to add a case):**
`tests/external/README.md`.

## CI / pre-push

### Tiered gate

Tiers, fastest first. The contract: **CI mirrors every phase of `make test`, so
anything a faster local tier skips is still caught before merge** — `main` is
branch-protected and CI gates the merge.

- **`make test-fast`** (inner loop, faster): the whole Rust workspace (unit +
  integration + doctests) via `cargo test`. Skips OCaml `dune runtest`,
  `check-reactive-golden`, the cross-language `test-integration` suite, and the
  doc gates. Use it while iterating. NOT authoritative.
- **`make test`** (authoritative, slow): every phase. Run before a change lands,
  or let the pre-push hook / CI run it.
- **CI** (`.github/workflows/`): the same surface, split across parallel
  workflows (faster wall-clock than the sequential local `make test`), on every
  push to `main` and every PR.

What `test-fast` skips is still gated elsewhere: OCaml unit tests → Compiler
workflow; reactive/regression golden compile-drift → the golden-diff in CI + the
pre-push hook (which now diff `tests/fixtures/reactive/ir` and
`tests/fixtures/regression/ir`, not just `ir/golden`/`ocaml/golden`);
cross-language integration → `ci.yml`; doc / CLI-doc gates → Doctest / CLI-docs
workflows. Nothing the fast tier skips is un-gated.

**Pre-push hook (`.githooks/pre-push`, installed via `core.hooksPath`).**
Mirrors CI — runs locally on every `git push`:

1. OCaml build + tests
2. Rust `cargo test --workspace --no-fail-fast` (unit + integration + doctests)
3. `cargo clippy --all-targets -- -D warnings`
4. `make update-golden` + assert the golden corpora unchanged — `ir/golden/`,
   `ocaml/golden/`, **and** `tests/fixtures/reactive/ir/` +
   `tests/fixtures/regression/ir/` (which live outside the first two, so a
   reactive/regression compile-drift would otherwise slip through)
5. `make test-integration`

(The book used to be built here via `mdbook`; it now lives in `../camdl-book`
with its own CI. Remove any stale `.githooks/pre-push` entry that still invokes
`mdbook build` if present.)

**Bypass only for documentation-only commits with `--no-verify`.** Otherwise
never — see the comment at the top of the hook about the 2026-04-17 commit that
broke CI because `cargo check --tests` compiled tests without running them.

**GitHub Actions (`.github/workflows/`).** The full gate, split across parallel
workflows, on push to `main` and on PRs:

- `ci.yml` — clippy, `make test-rust` (workspace except sim; cargo test), the
  golden-diff (all four corpora), `make test-integration`
- `inference.yml` — `make build-benches` (compile-only build of every bench, so
  bench bit-rot fails fast — gh#222) then `make test-inference` (the sim crate;
  cargo test)
- `compiler.yml` — `make test-ocaml` (`dune runtest`)
- `doctest.yml` — `make test-docs` (camdlc doctest of the spec set)
- `cli-docs.yml` — `make test-cli-docs`
- `release.yml` — release artifacts (Linux / macOS / Windows)

Statistical `#[ignore]` tests are **not** in CI yet. Run manually before
releases (`cargo test --release --workspace -- --ignored`); nightly CI job
planned.

## Writing tests

### Adding a spec-claim regression

The 2026-04-21 table-unit incident was a spec claim nothing tested. Follow this
discipline for any spec claim that the compiler / runtime must uphold:

1. Write the test **before** the fix (TDD).
2. Confirm it fails against the unfixed code.
3. Fix.
4. Confirm the test now passes.
5. Commit both in the same change.

Example: `rust/crates/cli/tests/scenario_runtime_application.rs`,
`ocaml/test/test_compiler.ml::table_unit_conversion`.

The commit message should mention which spec section's claim the test guards
(§X.Y) so future drift has a breadcrumb.

### Adding an error-code fixture

For every `emit_error ctx ~code:"ENNN" …` in the compiler:

1. Create `ocaml/test/errors/ennn_<slug>.camdl` — a minimal model that triggers
   the error and nothing else.
2. Verify manually: `camdlc check ocaml/test/errors/ennn_<slug>.camdl` emits the
   expected code.
3. The `negative_golden` suite in `test_compiler.ml` picks up the fixture
   automatically — no glue code needed.

Coverage status (2026-04-21): 90 codes emitted, 26 tested. 64 have no fixture —
see `docs/dev/reviews/2026-04-21-spec-claims-vs-tests.md` §P2 for the list.

### Adding a statistical test

See `docs/dev/runtime-simulation-tests.md` for the full pattern. Skeleton:

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

Always include a "distinguishable from the degenerate case" sanity assertion
alongside the quantitative match — a regression that collapses to a
different-but-similar-mean distribution might slip through pointwise checks.

### Adding a golden fixture

1. Write the `.camdl` at `ocaml/golden/<name>.camdl`.
2. `make update-golden` — regenerates `<name>.ir.json`.
3. **Review the JSON diff before committing.** The golden is your ground truth;
   if the regeneration changed values you didn't expect, that's a compiler bug
   to investigate, not a "update and move on."
4. Optionally add `ocaml/golden/<name>.params.toml` for simulation tests that
   need parameter values.

The IR round-trip test (L2) and smoke-all-golden test (L4) will automatically
pick up the new fixture.

### Updating a golden intentionally

When a compiler change legitimately alters golden IR (e.g., the 2026-04-21
table-unit fix that changed `sir_five_age.ir.json` values from `[5.0, 10.0, …]`
to `[1826.2, 3652.4, …]`):

1. `make update-golden`
2. **Diff each regenerated file and check the values are the ones you
   intended.** The pre-push hook will now complain that the golden is dirty —
   that's working as designed.
3. Commit the compiler change and the golden update together.

## Gotchas

- **`#[ignore]`** on Rust tests means opt-in. Always run with `-- --ignored`
  before merging anything that touches the sim or inference code.
- **Deterministic backend for scenario / value tests.** Use `--backend ode` in
  shell-out tests whose assertion is on a scalar output. Stochastic backends
  (Gillespie, chain-binomial) introduce seed-dependent noise that can mask an
  off-by-one or a no-op.
- **`--release`.** Almost every test runs under release. Debug-build runs of the
  sim tests are ~10× slower; the statistical tests become unbearable. Exception:
  rapid iteration on a single test while you're debugging — `cargo test <name>`
  without `--release` is fine for a minute.
- **Parallel test execution.** `cargo test` uses threads by default. Tests that
  touch shared filesystem state (`tempdir()` is fine; `/tmp` is not) can race.
  If you see flakiness, pass `-- --test-threads=1`.
- **Golden-file drift.** The pre-push hook runs `make update-golden` and checks
  for dirty working tree. If a schema change requires updates, update + commit
  in the same branch.
- **camdlc version pin.** The `camdl` binary refuses to run against a mismatched
  `camdlc`. `make test` handles this itself (PATH-prepends the fresh camdlc in
  `test-rust`) — see "Gotcha: camdlc version check" above. For ad-hoc `camdl`
  runs, `CAMDL_SKIP_VERSION_CHECK=1` or
  `CAMDLC=ocaml/_build/default/bin/camdlc.exe`.

## When tests disagree with each other

If L4 (Rust unit) passes but L5 (CLI shell-out) fails, the bug is in the CLI
glue — arg parsing, path resolution, or the util-layer model mutation (params
application, scenario filter). If L5 passes but L7 fails, the bug is
cross-language — OCaml emits something Rust doesn't understand, or vice versa.
If L1-L4 pass but L6 fails, the compiler produces correct-looking IR whose
runtime dynamics are wrong (the 2026-04-21 table-unit bug was exactly this shape
— compiler test passed, no runtime check existed). Use the layer disagreement to
triangulate.
