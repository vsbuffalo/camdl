---
date: 2026-11-21
status: implemented
related: gh#227
---

# Optimizing the inference crates in the test profile

## Question

`cargo test` builds the dev profile, at opt-level 0. The `sim` crate's tests run
particle filters, conditional SMC and MCMC for thousands of iterations, and on
CI four of its 188 test binaries took 661 of the 856 s the `sim` suite spent
running tests. Does compiling `sim`, `numerics`, `score` and the third-party
crates at opt-level 2, with debug assertions and overflow checks left on, change
any test outcome, and what does it cost and save?

## Setup

A worktree at `9ae418855`, on a 16-core M4 Max shared with three other agents'
`make test` runs (load averages 13–54 during the measurements, noted beside
each). Both runs used the gate's settings: `RAYON_NUM_THREADS=2`,
`CARGO_BUILD_JOBS=6`, `RUST_TEST_THREADS=4`, the freshly built `camdlc` first on
`PATH`. The optimized run passed the same four
`--config 'profile.dev.package.<p>.opt-level=2'` overrides that
`rust/Cargo.toml` now declares.

## Measurements

The whole `sim` suite, `cargo test --no-fail-fast -p sim`:

| Build       | Compile | Run   | Load during run | Passed | Failed | Ignored |
| ----------- | ------- | ----- | --------------- | ------ | ------ | ------- |
| opt-level 0 | 36 s    | 592 s | 13–18           | 1,480  | 0      | 30      |
| opt-level 2 | 158 s   | 132 s | 21–27           | 1,480  | 0      | 30      |

The sorted list of 1,484 `test <name> ... <outcome>` lines is identical between
the two runs (`diff` empty). The optimized compile includes building every
third-party dependency optimized from nothing; a worktree pays that once.

Two of the slowest binaries, run alone earlier the same morning:

| Binary                     | opt-level 0 | opt-level 2 |
| -------------------------- | ----------- | ----------- |
| `pf_exact_marginal_oracle` | 129.6 s     | 11.4 s      |
| `pmmh`                     | 122.6 s     | 10.0 s      |

Rebuilding the `sim` tests after a one-line edit inside a function body of
`sim/src/inference/resampling.rs` (`cargo test -p sim --no-run`), twice each:

| Build       | Rebuild    | Load  |
| ----------- | ---------- | ----- |
| opt-level 0 | 26 s, 25 s | 20–44 |
| opt-level 2 | 27 s, 40 s | 16–30 |

## Interpretation

No test outcome changed, which is what the language guarantees: rustc emits no
fast-math flags, so LLVM neither reorders nor contracts floating-point
operations at any opt-level, and the tests that pin exact trajectories and
inference results (`gate_inference_baseline`, `gate_trajectory_baseline`, the
determinism tests) passed under both builds.

The suite runs about 4.5 times faster (592 s to 132 s), under a heavier load. An
edit to `sim` rebuilds in roughly the same time, within the noise of a shared
machine; `cli` and the other workspace crates are not optimized, so edits there
rebuild exactly as before.

## Next

The cost that remains is linking and launching 452 integration-test binaries
(gh#227); a timing report of a full gate will say how large it is now.
