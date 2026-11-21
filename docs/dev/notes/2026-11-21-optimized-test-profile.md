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

## Result on an idle machine

The full gate, `make test`, at the commit before the change (`9ae418855`) and
with it (`c2ff7b0d5`), each run once with `camdl-cloud test --timing` on a
Google Batch machine of its own: `c4a-highcpu-8` (8 Arm vCPUs, 16 GB,
linux/arm64), cold caches, the local gate's settings (`CARGO_BUILD_JOBS=6`,
`RUST_TEST_THREADS=4`, `RAYON_NUM_THREADS=2`), nothing else running. Both
passed. Times are from the gate-timing log (`scripts/timed_shell.py`); the runs
are `gate-base-1` and `gate-speed-1`, and `camdl-cloud pull <run>` retrieves
their `gate.log`, `timing.tsv`, `timing-report.txt` and `vmstat.txt`.

| Phase                          | opt-level 0 | opt-level 2 | Change |
| ------------------------------ | ----------- | ----------- | ------ |
| `sim` tests: compile           | 120 s       | 224 s       | +104 s |
| `sim` tests: run               | 955 s       | 111 s       | −844 s |
| rest of the workspace: compile | 86 s        | 166 s       | +80 s  |
| rest of the workspace: run     | 738 s       | 704 s       | −34 s  |
| `cargo build --release`        | 118 s       | 123 s       | +5 s   |
| `cargo clippy`                 | 40 s        | 64 s        | +24 s  |
| Whole gate                     | 2,098 s     | 1,433 s     | −665 s |

Time inside the `sim` test harnesses fell 8.8-fold (`pmmh` 234 s to 14 s,
`pf_exact_marginal_oracle` 200 s to 14 s). The optimized build adds about 184 s
of compile from cold caches; a worktree or CI cache pays it once. The clippy
difference is unexplained from one run each.

Two properties of the gate itself, from the same runs:

- **Launching test binaries costs nothing on Linux.** The wall time from cargo's
  `Running` line to each binary's last result, less its harness time, totals 1–2
  s over 482 binaries. A timed gate on the shared M4 Max counted 838 s for the
  same quantity, so that cost is macOS's first-launch assessment and the Mac's
  load, not cargo or the number of binaries.
- **The test phases leave most of the machine idle.** From `vmstat` samples
  every 5 s, CPU is 27–33 % busy while tests run (one binary at a time, at most
  four test threads) and 60–74 % busy while compiling (six jobs). The phases run
  one after another. Free memory reaches 0.1 GB while the test binaries link, on
  16 GB.

## Setup of the shared-machine measurements

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

On Linux the gate's remaining time is the tests themselves, run serially on a
mostly idle machine. Running the `sim` suite and the rest of the workspace at
the same time would end the gate near the end of the longer one, if memory
allows both to link at once. The longest single binary is now
`external_validation` (one test, 160 s). Consolidating test binaries (gh#227)
would save time only on macOS.
