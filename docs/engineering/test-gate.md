# The test gate

`make test` is the check every change passes before it lands on `main`, and this
page describes how it runs: what it runs, how several gates share one machine,
how a gate that already passed is not run again, and where its time goes. It is
for a developer changing the Makefile's test targets, the pre-push hook or the
test layout. The reference for each piece, and the test tiers, is
[Testing camdl](../dev/testing.md).

## What the gate runs

`make test` runs these phases one after another. Each is also a target of its
own.

| Phase                        | Target                              | What it checks                                                                          |
| ---------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------- |
| Lint                         | `lint`                              | `cargo clippy --workspace --all-targets -D warnings`                                    |
| Compiler                     | `test-ocaml`, the `check-*-golden`s | the OCaml compiler's tests, and that the committed golden IR files still compile as is  |
| Release build                | `build-rust`                        | the `camdl` binary that tests which start `camdl` as a process use                      |
| The rest of the workspace    | `test-rust`                         | `cargo test --workspace --exclude sim`: mostly the `cli` crate, about 300 test binaries |
| The simulation and inference | `test-inference`                    | `cargo test -p sim`: the engine and the inference stack, about 190 test binaries        |
| Cross-language               | `test-integration`                  | OCaml-compiled models run by the Rust runtime                                           |
| Documentation and the rest   | `test-docs`, `test-cli-docs`, …     | the language specification's examples, the CLI pages, the install script, these scripts |

`cargo test` compiles each file in a crate's `tests/` directory into its own
executable, a test binary, and runs the binaries one at a time; within a binary
the tests run in parallel on `RUST_TEST_THREADS` threads. A binary holding one
long test therefore runs on one thread, however many cores the machine has.

The test profile compiles `sim`, `numerics`, `score` and every third-party crate
at opt-level 2, with debug assertions and overflow checks on (`rust/Cargo.toml`,
`[profile.dev.package.*]`). The particle-filter, CSMC and MCMC tests run
thousands of iterations, and unoptimized they took 955 s of a 2,098 s gate;
optimized they take about 100 s, with every test outcome unchanged
([the measurement](../dev/notes/2026-11-21-optimized-test-profile.md)).

## Sharing a machine

Several checkouts of the repository on one machine, such as agents in separate
worktrees, can each start a gate. Unqueued, seven gates at once drove a 16-core
machine to a load of about 145, and every gate slowed. Two mechanisms divide the
machine: a queue limits how many gates run at once, and each gate sizes its
threads to its share.

**The queue.** `scripts/test-slot.sh` is a counting semaphore: `make test`,
`make test-rust`, `make test-inference`, `make test-integration` and the
pre-push hook each hold one of `CAMDL_TEST_SLOTS` slots while they run, and wait
while all are taken. A slot is an exclusive `flock(2)` lock on one of the files
`slot.0` … `slot.N-1` in a directory every session on the machine shares, so the
kernel releases it when the holder exits, however it exits. A gate takes one
slot for all its phases: the queued target runs `make <target>` again inside its
slot with `CAMDL_TEST_SLOT_HELD=1` exported, and under that marker every queued
target runs straight through. Waiters retry every two seconds, in no particular
order.

**The share.** Each gate uses the machine's cores divided by `CAMDL_TEST_SLOTS`
(all of them when the queue is off): that many compile jobs, that many test
threads per test binary, and half as many Rayon threads in each test binary and
in each `camdl` a test starts. Rayon is the library camdl's parallel loops use;
the Rayon count is half the share because a binary running several tests at once
starts one Rayon pool per test, and at the full share those binaries ran slower
from the extra threads.

| Machine                            | `CAMDL_TEST_SLOTS`        | Compile jobs | Test threads | Rayon threads |
| ---------------------------------- | ------------------------- | ------------ | ------------ | ------------- |
| A laptop with one checkout         | unset (1)                 | all cores    | all cores    | half          |
| A 16-core machine shared by agents | 3, set in its environment | 5            | 5            | 2             |
| An 8-vCPU cloud machine            | unset (1)                 | 8            | 8            | 4             |
| CI, one job per runner             | 0 (queue off)             | all cores    | all cores    | half          |

The default is one slot, so a machine with one checkout gives its gate the whole
machine. A machine meant to run several gates at once sets the slot count in the
environment of every shell that starts one; in zsh, the line
`export CAMDL_TEST_SLOTS=3` goes in `~/.zshenv`, which non-interactive shells
also read. An explicit `CARGO_BUILD_JOBS`, `RUST_TEST_THREADS` or
`RAYON_NUM_THREADS` in the environment overrides its share.

## Not running a gate twice

The pre-push hook runs a subset of `make test`, so a push straight after a
passing gate would repeat work. A passing `make test` therefore records the git
tree hash it tested, the hash of every tracked file's path and content, as a
file in `<git common dir>/camdl-gate-passed/`, shared by every worktree of the
repository. The hook exits at once when every pushed commit's tree has a record.
A record is written only when the checkout had no uncommitted change, and no
untracked file under a directory the gate reads, from the start of the gate to
its end. `scripts/gate-stamp.sh` keeps the records, and `make test-gate-stamp`
tests its rules.

The same idea, keying a test result by a hash of exactly what was tested, is how
[Bazel](https://bazel.build/reference/command-line-reference#flag--cache_test_results)
caches test results; the tree hash is the hash git already keeps.

A passed run of the whole gate on camdl-cloud counts as a record too.
camdl-cloud is a separate tool, used by the maintainers, that runs a command on
a Google Cloud machine of its own. `camdl-cloud test --rev <commit>` clones that
one commit on a fresh machine, refuses to run unless the clone's tree is clean,
runs `make test`, and records the outcome. The hook reads those records with
`camdl-cloud runs --json` and accepts a pushed commit whose tree is the tree of
a run with target `test` and outcome `passed`. Without `camdl-cloud` on `PATH`,
or with `CAMDL_GATE_CLOUD=0`, only local records count. A record covers the
repository's content, not the state outside it, such as the installed opam
packages, so CI still runs the full gate on every push and pull request.

With a cloud pass, landing a change needs no gate on the developer's machine:
commit, `camdl-cloud test --rev HEAD`, wait for `passed`, push. The cloud
machine runs linux/arm64 and CI runs linux/x86-64, so a failure that occurs only
on macOS is found by the next local gate rather than before the push.

## Where the time goes

The whole gate at commit `617e10afe` on an otherwise idle 8-vCPU machine
(`c4a-highcpu-8`, 16 GB, linux/arm64), from cold caches, timed with
`camdl-cloud test --timing` (run `gate-share-2`); repeated runs of one commit
agree to about 20 s.

| Phase                                 | Time    | CPU busy |
| ------------------------------------- | ------- | -------- |
| The rest of the workspace: running    | 505 s   | ~60 %    |
| The simulation and inference: compile | 177 s   | ~95 %    |
| The rest of the workspace: compile    | 134 s   | ~90 %    |
| The simulation and inference: running | 98 s    | ~70 %    |
| Release build                         | 97 s    | ~80 %    |
| Lint                                  | 56 s    | ~65 %    |
| Everything else                       | ~45 s   |          |
| Whole gate                            | 1,114 s |          |

The same commit's gate at the earlier fixed settings (six compile jobs, four
test threads, two Rayon threads; run `gate-main-3` at `bd2717e98`) took 1,433 s,
and before the optimized test profile (run `gate-base-1` at `9ae418855`) 2,098
s.

Launching a test binary costs nothing measurable on Linux: over 482 binaries,
the time from cargo starting a binary to its first test, less the time the
binary reports for its tests, totals 1–2 s. On macOS the first launch of a newly
linked binary waits for the system's security assessment, which measured 838 s
over one gate on a loaded 16-core machine. That is why the gate does not use
`cargo-nextest`: it starts many test binaries at once, and that burst of first
launches stalled the assessment on macOS.

`make gate-timing` times a local gate the same way and prints where it went:
each phase, each cargo invocation split into compile, test and launch time, and
each test binary.
