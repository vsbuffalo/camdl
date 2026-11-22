---
date: 2026-11-21
status: implemented
related: docs/engineering/test-gate.md
---

# Sizing each gate's threads from the machine

## Question

The gate ran with fixed limits chosen for one 16-core machine shared by three
gates: six compile jobs, four test threads per test binary (from
`rust/.cargo/config.toml`) and two Rayon threads. On a machine running one gate
those limits leave cores idle: on an 8-vCPU machine the CPU was 27–33 % busy
while tests ran. How much does sizing the three numbers to the machine's cores
divided by the test slots save, and should Rayon get the whole share or half?

## Setup

`camdl-cloud test --timing`: the whole `make test` on an otherwise idle
`c4a-highcpu-8` (8 vCPUs, 16 GB, linux/arm64), cold caches, one run each. All
three runs are of commit `bd2717e98` or a commit adding only the Makefile change
to it. `vmstat` sampled the machine every 5 s.

## Measurements

| Run            | Commit      | Jobs / test threads / Rayon | Whole gate | Rest of the workspace: run | `sim`: run | `external_validation` |
| -------------- | ----------- | --------------------------- | ---------- | -------------------------- | ---------- | --------------------- |
| `gate-main-3`  | `bd2717e98` | 6 / 4 / 2                   | 1,433 s    | 706 s                      | 107 s      | 160 s                 |
| `gate-share-1` | `b2b523d4c` | 8 / 8 / 8                   | 1,095 s    | 466 s                      | 118 s      | 43 s                  |
| `gate-share-2` | `617e10afe` | 8 / 8 / 4                   | 1,114 s    | 505 s                      | 98 s       | 82 s                  |

All three passed. Compiles shortened too: the test binaries from 389 s to
311–312 s, the release build from 120 s to 97 s. CPU while the rest of the
workspace's tests ran rose from 33 % to 59 % busy. Two repeat runs of one
configuration on other commits differed by 22 s overall.

## Interpretation

Sizing to the machine saves 320–340 s of 1,433 s. The two Rayon settings are 19
s apart, within the run-to-run difference. The full share speeds the single-test
`external_validation` (ten particle-filter replicates in one `camdl` process);
half the share avoids slowing binaries that run several tests at once, each
starting its own Rayon pool (`zz_cond_systematic_probe` 23 s, 36 s and 25 s;
`csmc_exact_invariance` 24 s, 32 s and 25 s). Half is chosen for the machine
shared by three gates: there each gate's share is 5, and half gives 2 Rayon
threads, the setting it had, where the full share would put 25 threads on 5
cores.

Running the two cargo test suites at the same time instead (one sub-make at
`-j2`, run `gate-overlap-1`) took 2,902 s: make's two job tokens limited each
cargo to about one compile job, and done correctly it could save at most the
shorter suite's run, about 100 s.
